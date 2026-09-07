import io
import requests
import concurrent.futures
import numpy as np
from PIL import Image
from pystac_client import Client
import planetary_computer
from skimage import exposure

class SentinelFetcher:
    def __init__(self, mosaic_path=r"earth pyopengl texture\high res earth image texture.png"):
        """
        Initializes the STAC client and the Global Mosaic 'Source of Truth'.
        """
        self.api_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self.catalog = Client.open(self.api_url, modifier=planetary_computer.sign_inplace)
        
        # Connection pooling for high-speed multithreading
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.session.mount('https://', adapter)
        
        # Load Global Mosaic (4096x4096px)
        self.mosaic = Image.open(mosaic_path).convert("RGB")
        self.m_size = 4096

    def _get_mosaic_patch(self, bbox):
        """
        Crops a 512x512 reference from the 4k Global Mosaic based on GPS coordinates.
        This provides the 'target' colors for histogram matching.
        """
        minx, miny, maxx, maxy = bbox
        
        # Map Lat/Lon to 0.0 - 1.0 UV space
        u1 = (minx + 180.0) / 360.0
        v1 = (90.0 - maxy) / 180.0
        u2 = (maxx + 180.0) / 360.0
        v2 = (90.0 - miny) / 180.0
        
        # Convert to pixel coordinates on the 4k map
        left = int(max(0, u1 * self.m_size))
        top = int(max(0, v1 * self.m_size))
        right = int(min(self.m_size, u2 * self.m_size))
        bottom = int(min(self.m_size, v2 * self.m_size))
        
        # Crop and resize to match our 512px tile resolution
        patch = self.mosaic.crop((left, top, right, bottom))
        return np.array(patch.resize((512, 512), resample=Image.BILINEAR))

    def _download_layer(self, item_id: str, bbox: tuple) -> Image.Image | None:
        """Downloads a single 512x512 visual asset for the given bounding box."""
        minx, miny, maxx, maxy = bbox
        url = (
            f"https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/"
            f"{minx},{miny},{maxx},{maxy}.png?collection=sentinel-2-l2a"
            f"&item={item_id}&assets=visual&width=512&height=512"
        )
        try:
            # We must sign the URL so the Data API allows the download
            signed_url = planetary_computer.sign_url(url)
            response = self.session.get(signed_url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            return None

    def _match_histograms_numpy(self, source, reference):
        """Pure NumPy replacement for skimage.exposure.match_histograms"""
        matched = np.zeros_like(source)
        for channel in range(source.shape[-1]):
            s_chan = source[..., channel]
            r_chan = reference[..., channel]
            
            s_values, bin_idx, s_counts = np.unique(s_chan.ravel(), return_inverse=True, return_counts=True)
            r_values, r_counts = np.unique(r_chan.ravel(), return_counts=True)
            
            s_quantiles = np.cumsum(s_counts).astype(float) / s_chan.size
            r_quantiles = np.cumsum(r_counts).astype(float) / r_chan.size
            
            interp_values = np.interp(s_quantiles, r_quantiles, r_values)
            matched[..., channel] = interp_values[bin_idx].reshape(s_chan.shape)
        return matched.astype(np.uint8)

    def pull_tile(self, fetch_state: dict) -> dict | None:
        bbox = fetch_state["bbox"]
        try:
            search = self.catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox, 
                sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
                max_items=12 
            )
            items = list(search.items())
            if not items: return None

            raw_layers = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self._download_layer, it.id, bbox) for it in items]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res is not None: raw_layers.append(np.array(res))

            if not raw_layers: return None

            # --- ARTIFACT STRIPPING ---
            stack = np.stack(raw_layers, axis=0).astype(float)
            stack[np.all(stack < 1, axis=-1) | np.all(stack > 250, axis=-1)] = np.nan
            
            with np.errstate(all='ignore'):
                median_image = np.nanmedian(stack, axis=0)

            # --- DATA BLOCK FILL ---
            reference_patch = self._get_mosaic_patch(bbox)
            pixel_mask = np.all(np.isnan(median_image), axis=-1)
            if np.any(pixel_mask):
                median_image[pixel_mask] = reference_patch[pixel_mask].astype(float)

            # --- 5. RAW COLOR SYNC ---
            # 1. Convert to float for precise math
            median_image = np.nan_to_num(median_image, nan=0).astype(np.float32)
            reference_patch = reference_patch.astype(np.float32)

            # 2. Calculate the "Brightness Gap"
            # We find the average brightness of both
            avg_sentinel = np.mean(median_image)
            avg_reference = np.mean(reference_patch)

            # 3. Apply the Offset (Sentinel -> Reference)
            # If Sentinel is 200 and Reference is 100, we multiply by 0.5
            if avg_sentinel > 0:
                exposure_ratio = avg_reference / avg_sentinel
                # We cap the ratio so it doesn't do anything crazy in pitch black areas
                exposure_ratio = np.clip(exposure_ratio, 0.1, 1.0) 
                corrected_image = median_image * exposure_ratio
            else:
                corrected_image = median_image

            # 4. Final Hard Cap
            # We never let the raw data 'clip' at 255. We keep it under 230 
            # so it looks like a solid physical object, not a lightbulb.
            corrected_image = np.clip(corrected_image, 0, 230).astype(np.uint8)

            final_img = Image.fromarray(corrected_image)
            
            # Create an in-memory byte stream
            buf = io.BytesIO()
            final_img.save(buf, format="PNG", optimize=True)
            byte_im = buf.getvalue()

            return {
                #"pixels": final_img.convert("RGBA").tobytes(),
                "pixels": byte_im,
                "width": 512, "height": 512,
                "bbox": bbox, 
                "key": fetch_state.get("key"),
                "slot": fetch_state.get("slot")
            }
        except Exception as e:
            print(f"[SentinelFetcher] Composite Error: {e}")
            return None