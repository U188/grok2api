from typing import Any, List, Optional
from tempfile import SpooledTemporaryFile
from urllib.parse import urlparse
import aiohttp

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.api.v1.chat import _safe_sse_stream
from app.core.config import get_config
from app.core.exceptions import AppException, ErrorType, ValidationException
from app.services.grok.services.image import ImageGenerationService
from app.services.grok.services.image_edit import ImageEditService
from app.services.grok.services.model import ModelService
from app.services.token import get_token_manager

router = APIRouter(tags=["Images"])

IMAGINE_FAST_MODEL_ID = "grok-imagine-1.0-fast"
ALLOWED_IMAGE_SIZES = {
    "1280x720",
    "720x1280",
    "1792x1024",
    "1024x1792",
    "1024x1024",
}


def resolve_aspect_ratio(size: Optional[str]) -> str:
    aspect_ratio_map = {
        "1280x720": "16:9",
        "720x1280": "9:16",
        "1792x1024": "3:2",
        "1024x1792": "2:3",
        "1024x1024": "1:1",
    }
    return aspect_ratio_map.get(size or "1024x1024", "1:1")


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="生成提示词")
    model: str = Field(default="grok-imagine-1.0", description="模型名称")
    n: Optional[int] = Field(default=1, ge=1, le=10)
    size: Optional[str] = Field(default="1024x1024")
    response_format: Optional[str] = Field(default=None, description="url|b64_json|base64")


class ImageEditRequest(BaseModel):
    prompt: str = Field(..., description="编辑提示词")
    model: str = Field(default="grok-imagine-1.0-edit", description="模型名称")
    n: Optional[int] = Field(default=1, ge=1, le=10)
    size: Optional[str] = Field(default="1024x1024")
    response_format: Optional[str] = Field(default=None, description="url|b64_json|base64")
    user: Optional[str] = None


class ImagesResponse(BaseModel):
    created: int
    data: List[dict[str, Any]]


def _resolve_image_format(value: Optional[str]) -> str:
    fmt = value or get_config("app.image_format") or "url"
    if isinstance(fmt, str):
        fmt = fmt.lower()
    if fmt == "base64":
        return "b64_json"
    if fmt in ("b64_json", "url"):
        return fmt
    raise ValidationException(
        message="response_format must be one of url, base64, b64_json",
        param="response_format",
        code="invalid_image_format",
    )


def _image_field(response_format: str) -> str:
    return "url" if response_format == "url" else "b64_json"


def _validate_image_params(n: int, size: str, stream: bool = False):
    if n < 1 or n > 10:
        raise ValidationException(
            message="n must be between 1 and 10",
            param="n",
            code="invalid_n",
        )
    if size not in ALLOWED_IMAGE_SIZES:
        raise ValidationException(
            message=f"size must be one of: {', '.join(sorted(ALLOWED_IMAGE_SIZES))}",
            param="size",
            code="invalid_size",
        )
    if stream and n > 1:
        raise ValidationException(
            message="streaming image generation only supports n=1",
            param="n",
            code="invalid_n",
        )


async def _pick_token_for_model(model: str):
    token_mgr = await get_token_manager()
    await token_mgr.reload_if_stale()

    token = None
    for pool_name in ModelService.pool_candidates_for_model(model):
        token = token_mgr.get_token(pool_name)
        if token:
            break

    if not token:
        raise AppException(
            message="No available tokens. Please try again later.",
            error_type=ErrorType.RATE_LIMIT.value,
            code="rate_limit_exceeded",
            status_code=429,
        )

    return token_mgr, token


def _images_response(model: str, response_format: str, images: List[str]) -> dict:
    field = _image_field(response_format)
    data = [{field: image} for image in images]
    return {"created": 0, "data": data, "model": model}


@router.post("/images/generations", response_model=ImagesResponse)
async def generate_image(request: ImageGenerationRequest):
    try:
        model = request.model or "grok-imagine-1.0"
        model_info = ModelService.get(model)
        if not model_info or not model_info.is_image:
            raise ValidationException(
                message="model does not support image generation",
                param="model",
                code="invalid_model",
            )

        n = request.n or 1
        size = request.size or "1024x1024"
        _validate_image_params(n, size, stream=False)
        response_format = _resolve_image_format(request.response_format)
        aspect_ratio_map = {
            "1280x720": "16:9",
            "720x1280": "9:16",
            "1792x1024": "3:2",
            "1024x1792": "2:3",
            "1024x1024": "1:1",
        }
        aspect_ratio = aspect_ratio_map.get(size, "1:1")

        token_mgr, token = await _pick_token_for_model(model)
        result = await ImageGenerationService().generate(
            token_mgr=token_mgr,
            token=token,
            model_info=model_info,
            prompt=request.prompt,
            n=n,
            response_format=response_format,
            size=size,
            aspect_ratio=aspect_ratio,
            stream=False,
            chat_format=False,
        )
        return _images_response(model, response_format, result.data)
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=e.message)
    except AppException as e:
        raise HTTPException(status_code=e.status_code or 500, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/edits")
async def edit_image(request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        image_inputs: List[str] = []
        prompt: Optional[str] = None
        model: str = "grok-imagine-1.0-edit"
        n: int = 1
        response_format: Optional[str] = None

        if content_type.startswith("application/json"):
            body = await request.json()
            prompt = body.get("prompt")
            model = body.get("model") or model
            n = int(body.get("n") or 1)
            response_format = body.get("response_format")

            image_urls = body.get("image_url")
            if not image_urls:
                raise HTTPException(status_code=400, detail="image_url is required for image edits")
            if isinstance(image_urls, str):
                image_urls = [image_urls]
            if not isinstance(image_urls, list):
                raise HTTPException(status_code=400, detail="image_url must be a string or array")
            image_inputs = [u for u in image_urls if isinstance(u, str) and u.strip()]
        else:
            form = await request.form()
            prompt = form.get("prompt")
            model = form.get("model") or model
            n_value = form.get("n")
            n = int(n_value) if n_value not in (None, "") else 1
            response_format = form.get("response_format")

            files = [item for item in form.getlist("image") if isinstance(item, UploadFile)]
            if not files:
                raise HTTPException(status_code=400, detail="image is required for image edits")

            for upload in files:
                raw = await upload.read()
                if not raw:
                    continue
                filename = upload.filename or "image"
                content_type_header = upload.content_type or "application/octet-stream"
                import base64
                encoded = base64.b64encode(raw).decode("ascii")
                image_inputs.append(f"data:{content_type_header};base64,{encoded}")

        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        model_info = ModelService.get(model)
        if not model_info or not model_info.is_image_edit:
            raise HTTPException(status_code=400, detail="model does not support image edits")

        response_format = _resolve_image_format(response_format)
        _validate_image_params(n, "1024x1024", stream=False)

        token_mgr, token = await _pick_token_for_model(model)
        result = await ImageEditService().edit(
            token_mgr=token_mgr,
            token=token,
            model_info=model_info,
            prompt=prompt,
            images=image_inputs,
            n=n,
            response_format=response_format,
            stream=False,
            chat_format=False,
        )
        return _images_response(model, response_format, result.data)
    except HTTPException:
        raise
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=e.message)
    except AppException as e:
        raise HTTPException(status_code=e.status_code or 500, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/variations")
async def create_image_variation(request: Request):
    raise HTTPException(status_code=501, detail="/v1/images/variations is not implemented")
