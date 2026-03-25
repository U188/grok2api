from typing import List, Optional
from tempfile import SpooledTemporaryFile
from urllib.parse import urlparse
import httpx

from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile

from app.models.image import (
    ImageGenerationRequest,
    ImageEditRequest,
    ImageVariationRequest,
    ImagesResponse,
)
from app.services.grok.grok_service import GrokService
from app.api.deps import get_grok_service

router = APIRouter()


@router.post("/images/generations", response_model=ImagesResponse)
async def generate_image(
    request: ImageGenerationRequest,
    grok_service: GrokService = Depends(get_grok_service),
):
    """
    Generate images from text prompts.

    Compatible with OpenAI's /v1/images/generations endpoint.
    """
    try:
        return await grok_service.generate_image(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/edits")
async def edit_image(
    request: Request,
    grok_service: GrokService = Depends(get_grok_service),
):
    """
    Edit images based on text prompts.

    Compatible with OpenAI's /v1/images/edits endpoint.
    Supports both multipart/form-data uploads and JSON body with image_url.
    """
    try:
        content_type = request.headers.get("content-type", "")
        image_files: Optional[List[UploadFile]] = None
        prompt: Optional[str] = None
        model: Optional[str] = None
        n: Optional[int] = None
        size: Optional[str] = None
        quality: Optional[str] = None
        user: Optional[str] = None

        if content_type.startswith("application/json"):
            body = await request.json()
            prompt = body.get("prompt")
            model = body.get("model")
            n = body.get("n")
            size = body.get("size")
            quality = body.get("quality")
            user = body.get("user")

            image_urls = body.get("image_url")
            if not image_urls:
                raise HTTPException(status_code=400, detail="image_url is required for image edits")
            if isinstance(image_urls, str):
                image_urls = [image_urls]
            if not isinstance(image_urls, list):
                raise HTTPException(status_code=400, detail="image_url must be a string or array")

            image_files = []
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                for idx, image_url in enumerate(image_urls, 1):
                    if not isinstance(image_url, str) or not image_url.strip():
                        raise HTTPException(status_code=400, detail=f"image_url[{idx}] is invalid")
                    try:
                        response = await client.get(image_url)
                        response.raise_for_status()
                    except Exception as exc:
                        raise HTTPException(status_code=400, detail=f"failed to download image_url[{idx}]: {exc}")

                    filename = f"image-{idx}"
                    parsed = urlparse(image_url)
                    if parsed.path and parsed.path.rsplit('/', 1)[-1]:
                        filename = parsed.path.rsplit('/', 1)[-1]

                    content_type_header = response.headers.get("content-type", "application/octet-stream")
                    tmp = SpooledTemporaryFile()
                    tmp.write(response.content)
                    tmp.seek(0)
                    image_files.append(
                        UploadFile(
                            file=tmp,
                            filename=filename,
                            headers={"content-type": content_type_header},
                        )
                    )
        else:
            form = await request.form()
            prompt = form.get("prompt")
            model = form.get("model")
            size = form.get("size")
            quality = form.get("quality")
            user = form.get("user")
            n_value = form.get("n")
            n = int(n_value) if n_value not in (None, "") else None

            image_files = [item for item in form.getlist("image") if isinstance(item, UploadFile)]

        request_data = ImageEditRequest(
            prompt=prompt,
            images=image_files,
            model=model,
            n=n,
            size=size,
            quality=quality,
            user=user,
        )
        return await grok_service.edit_image(request_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/variations", response_model=ImagesResponse)
async def create_image_variation(
    image: UploadFile,
    model: Optional[str] = None,
    n: Optional[int] = None,
    size: Optional[str] = None,
    user: Optional[str] = None,
    grok_service: GrokService = Depends(get_grok_service),
):
    """
    Create variations of an image.

    Compatible with OpenAI's /v1/images/variations endpoint.
    """
    try:
        request_data = ImageVariationRequest(
            image=image,
            model=model,
            n=n,
            size=size,
            user=user,
        )
        return await grok_service.create_image_variation(request_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
