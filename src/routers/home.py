from fastapi import APIRouter, status
from typing import Dict

router = APIRouter(
        tags=['home']
)


@router.get("/", status_code=status.HTTP_200_OK)
async def home() -> Dict[str, str]:
    return {"message": "Hello world"}


@router.get("/healthy", status_code=status.HTTP_200_OK)
async def healthy() -> Dict[str, str]:
    return {"status": "healthy"}
