# forecasting/dataset/metaculus.py
from dataclasses import dataclass
from typing import List, Optional, Set, AsyncIterator, Any, Dict, Iterator
from datetime import datetime
import httpx
import logging
from inspect_ai.dataset import Dataset, Sample

logger = logging.getLogger(__name__)
API_BASE_URL = "https://www.metaculus.com/api/posts/"

@dataclass
class MetaculusConfig:
    """Configuration for Metaculus dataset."""
    # Field selection
    input_fields: Set[str] = None
    target_fields: Set[str] = None
    
    # API configuration
    api_key: Optional[str] = None
    api_base_url: str = API_BASE_URL
    
    # API filtering options
    tournaments: Optional[List[str]] = None
    statuses: Optional[List[str]] = None
    forecaster_id: Optional[int] = None
    not_forecaster_id: Optional[int] = None
    open_time__gt: Optional[datetime] = None
    published_at__gt: Optional[datetime] = None
    scheduled_resolve_time__gt: Optional[datetime] = None
    forecast_type: Optional[List[str]] = None
    order_by: Optional[str] = None
    with_cp: bool = True

    def __post_init__(self):
        if self.statuses is None:
            self.statuses = ["resolved"]
            
        if self.input_fields is None:
            self.input_fields = {
                "id", "title", "description", "created_at", "open_time",
                "type", "options", "possibilities", "resolution_criteria",
                "fine_print", "label", "scaling", "published_at", 
                "projects", "author_id", "author_username"
            }
            
        if self.target_fields is None:
            self.target_fields = {
                "id", "title", "description", "created_at", "open_time",
                "type", "options", "possibilities", "resolution_criteria",
                "fine_print", "label", "scaling", "resolution", 
                "resolution_set_time", "actual_close_time",
                "actual_resolve_time", "status", "author_id", 
                "author_username", "published_at", "comment_count", 
                "nr_forecasters", "forecasts_count", "projects", 
                "vote", "aggregations"
            }

class MetaculusDataset(Dataset):
    """Dataset that lazily loads from Metaculus API."""
    
    def __init__(self, config: MetaculusConfig):
        self.config = config
        self._client = None
        self._next_url = config.api_base_url
        self._current_page: List[Dict] = []
        self._current_idx = 0
    
    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient()
            if self.config.api_key:
                self._client.headers["Authorization"] = f"Token {self.config.api_key}"
    
    async def _fetch_next_page(self) -> bool:
        """Fetch next page of results. Returns False if no more pages."""
        if not self._next_url:
            return False
            
        await self._ensure_client()
        
        params = {
            "tournaments": self.config.tournaments,
            "statuses": self.config.statuses,
            "forecaster_id": self.config.forecaster_id,
            "not_forecaster_id": self.config.not_forecaster_id,
            "open_time__gt": self.config.open_time__gt.isoformat() if self.config.open_time__gt else None,
            "published_at__gt": self.config.published_at__gt.isoformat() if self.config.published_at__gt else None,
            "scheduled_resolve_time__gt": self.config.scheduled_resolve_time__gt.isoformat() if self.config.scheduled_resolve_time__gt else None,
            "forecast_type": self.config.forecast_type,
            "order_by": self.config.order_by,
            "with_cp": self.config.with_cp
        }
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            response = await self._client.get(self._next_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            self._current_page = data["results"]
            self._next_url = data["next"]
            self._current_idx = 0
            
            logger.info(f"Fetched page with {len(self._current_page)} posts")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching posts: {str(e)}")
            return False
    
    def _post_to_sample(self, post: Dict[str, Any]) -> Sample:
        """Convert API post to Sample"""
        input_data = {}
        target_data = {}
        
        for field in self.config.input_fields:
            if field in post:
                input_data[field] = post[field]
            elif field in post["question"]:
                input_data[field] = post["question"][field]
                
        for field in self.config.target_fields:
            if field in post:
                target_data[field] = post[field]
            elif field in post["question"]:
                target_data[field] = post["question"][field]
        
        return Sample(input=input_data, target=target_data)
    
    async def __aiter__(self) -> AsyncIterator[Sample]:
        while True:
            # If we've exhausted current page, try to fetch next
            if self._current_idx >= len(self._current_page):
                if not await self._fetch_next_page():
                    break
            
            # Yield next sample from current page
            if self._current_idx < len(self._current_page):
                yield self._post_to_sample(self._current_page[self._current_idx])
                self._current_idx += 1
    
    def __iter__(self) -> Iterator[Sample]:
        raise NotImplementedError("Use async iteration with MetaculusDataset")
    
    def __len__(self) -> int:
        raise NotImplementedError("Length not known until iteration complete")
    
    def __getitem__(self, idx: int) -> Sample:
        raise NotImplementedError("Random access not supported for MetaculusDataset")