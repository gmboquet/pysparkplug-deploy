"""Minimal Vast.ai REST client.

Wraps the documented API (https://console.vast.ai/api/v0, Bearer auth): search offers, create an
instance from an offer, read instance status (ssh host/port), and destroy it. Just the calls the
training launcher needs — no SDK dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import httpx

API = "https://console.vast.ai/api/v0"


class VastError(Exception):
    pass


@dataclass
class Offer:
    id: int
    gpu_name: str
    num_gpus: int
    price: float  # $/hr (dph_total)
    raw: dict = field(default_factory=dict)


class VastClient:
    def __init__(self, api_key: str, base: str = API, timeout: float = 30.0) -> None:
        if not api_key:
            raise VastError("a vast.ai API key is required (set MIXLE_VAST_API_KEY)")
        self.base = base.rstrip("/")
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    def _request(self, method: str, path: str, **kw) -> dict:
        try:
            r = httpx.request(method, f"{self.base}{path}", headers=self._headers, timeout=self.timeout, **kw)
        except httpx.HTTPError as e:
            raise VastError(f"vast.ai request failed ({method} {path}): {e}") from e
        if r.status_code >= 400:
            raise VastError(f"vast.ai {method} {path} -> {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError:
            return {}

    @staticmethod
    def build_query(*, gpu_name: str | None, num_gpus: int, max_price: float | None, limit: int) -> dict:
        """The offer-search filter body (exposed for testing/dry-run)."""
        q: dict = {
            "rentable": {"eq": True},
            "num_gpus": {"gte": num_gpus},
            "type": "ondemand",
            "order": [["dph_total", "asc"]],
            "limit": limit,
        }
        if gpu_name:
            q["gpu_name"] = {"in": [gpu_name.replace("_", " ")]}
        if max_price is not None:
            q["dph_total"] = {"lte": max_price}
        return q

    def search_offers(
        self, *, gpu_name: str | None = None, num_gpus: int = 1, max_price: float | None = None, limit: int = 20
    ) -> list[Offer]:
        data = self._request("POST", "/bundles/", json=self.build_query(
            gpu_name=gpu_name, num_gpus=num_gpus, max_price=max_price, limit=limit))
        offers: list[Offer] = []
        for o in data.get("offers", []):
            offers.append(
                Offer(
                    id=int(o.get("id", 0)),
                    gpu_name=str(o.get("gpu_name", "")),
                    num_gpus=int(o.get("num_gpus", 0)),
                    price=float(o.get("dph_total") or 0.0),
                    raw=o,
                )
            )
        return offers

    def create_instance(
        self,
        offer_id: int,
        *,
        image: str,
        disk: int,
        onstart: str | None = None,
        runtype: str = "ssh_direct",
        env: dict | None = None,
        label: str | None = None,
    ) -> int:
        body: dict = {"client_id": "me", "image": image, "disk": disk, "runtype": runtype}
        if onstart:
            body["onstart"] = onstart
        if env:
            body["env"] = env
        if label:
            body["label"] = label
        data = self._request("PUT", f"/asks/{offer_id}/", json=body)
        cid = data.get("new_contract")
        if not cid:
            raise VastError(f"vast.ai did not return a new instance id: {data}")
        return int(cid)

    def instance(self, instance_id: int) -> dict:
        data = self._request("GET", f"/instances/{instance_id}/")
        inst = data.get("instances", data)
        if isinstance(inst, list):
            for i in inst:
                if str(i.get("id")) == str(instance_id):
                    return i
            return {}
        return inst if isinstance(inst, dict) else {}

    def destroy(self, instance_id: int) -> None:
        self._request("DELETE", f"/instances/{instance_id}/")
