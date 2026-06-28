"""
Image Verification Module for TruthShield v2.

Provides OCR text extraction, image manipulation detection, and
optional reverse image search via Google Cloud Vision.
"""

import os
import io
import struct
from typing import Dict, List, Optional, Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


class ImageVerifier:
    """Analyze images for text content, manipulation, and provenance.

    Args:
        google_api_key: Optional Google Cloud Vision API key for
            reverse image search.
    """

    # Software strings that indicate editing
    _EDIT_SOFTWARE = [
        "photoshop", "gimp", "paint.net", "affinity", "pixlr",
        "lightroom", "capture one", "darktable", "corel",
        "snapseed", "canva", "fotor", "befunky",
    ]

    # OCR languages — EasyOCR codes
    _OCR_LANGS = ["en", "hi", "bn", "ta", "mr"]

    def __init__(self, google_api_key: Optional[str] = None) -> None:
        self._google_api_key = google_api_key
        self._ocr_reader = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_image(
        self,
        image: Union[str, bytes],
        *,
        run_ocr: bool = True,
        check_manipulation: bool = True,
        reverse_search: bool = False,
    ) -> Dict:
        """Run full image verification pipeline.

        Args:
            image: File path (str) or raw image bytes.
            run_ocr: Extract text via OCR.
            check_manipulation: Run manipulation detection.
            reverse_search: Query Google Vision for reverse image search.

        Returns:
            Dict with ``ocr``, ``manipulation``, ``reverse_search``,
            ``metadata`` sub-dicts.
        """
        img_bytes = self._to_bytes(image)
        result: Dict = {"ocr": None, "manipulation": None, "reverse_search": None, "metadata": {}}

        if run_ocr:
            result["ocr"] = self.extract_text(img_bytes)
        if check_manipulation:
            result["manipulation"] = self.detect_manipulation(img_bytes)
        if reverse_search and self._google_api_key:
            result["reverse_search"] = self.reverse_image_search(img_bytes)

        return result

    def extract_text(self, image: Union[str, bytes]) -> Dict:
        """Extract text from an image using EasyOCR.

        Returns:
            Dict with ``text``, ``confidence``, ``word_count``,
            ``language_detected``.
        """
        img_bytes = self._to_bytes(image)
        try:
            reader = self._get_ocr_reader()
            if reader is None:
                return {"text": "", "confidence": 0.0, "word_count": 0,
                        "language_detected": "unknown", "error": "easyocr not available"}

            results = reader.readtext(img_bytes)
            texts = []
            confidences = []
            for (_, text, conf) in results:
                texts.append(text)
                confidences.append(conf)

            full_text = " ".join(texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            word_count = len(full_text.split())

            # Detect dominant language
            lang = "en"
            try:
                from langdetect import detect
                if full_text.strip():
                    lang = detect(full_text)
            except Exception:
                pass

            return {
                "text": full_text,
                "confidence": round(avg_conf, 3),
                "word_count": word_count,
                "language_detected": lang,
            }
        except Exception as e:
            return {"text": "", "confidence": 0.0, "word_count": 0,
                    "language_detected": "unknown", "error": str(e)}

    def detect_manipulation(self, image: Union[str, bytes]) -> Dict:
        """Detect signs of image manipulation.

        Uses EXIF metadata analysis and Error Level Analysis (ELA).

        Returns:
            Dict with ``manipulation_score`` (0-1),
            ``flags`` (list of strings), ``metadata_anomalies``.
        """
        img_bytes = self._to_bytes(image)
        flags: List[str] = []
        score = 0.0
        metadata_anomalies: List[str] = []

        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
        except ImportError:
            return {"manipulation_score": 0.0, "flags": [],
                    "metadata_anomalies": [], "error": "Pillow not available"}

        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            return {"manipulation_score": 0.0, "flags": [],
                    "metadata_anomalies": [], "error": f"Cannot open image: {e}"}

        # --- EXIF analysis ---
        try:
            exif_data = pil_img._getexif()
            if exif_data:
                exif_dict = {}
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    exif_dict[tag_name] = str(value)

                # Check for editing software
                software = exif_dict.get("Software", "").lower()
                for editor in self._EDIT_SOFTWARE:
                    if editor in software:
                        flags.append(f"Edited with {exif_dict.get('Software', 'unknown')}")
                        score += 0.3
                        break

                # Check for missing or suspicious dates
                date_orig = exif_dict.get("DateTimeOriginal")
                date_digi = exif_dict.get("DateTimeDigitized")
                date_mod = exif_dict.get("DateTime")
                if date_mod and date_orig and date_mod != date_orig:
                    metadata_anomalies.append("Modification date differs from original capture date")
                    score += 0.15
                if not date_orig and not date_digi:
                    metadata_anomalies.append("No original capture date in EXIF")
                    score += 0.05
            else:
                # No EXIF at all — could be stripped
                if pil_img.format in ("JPEG", "JPG"):
                    metadata_anomalies.append("JPEG with no EXIF data (may have been stripped)")
                    score += 0.1
        except Exception:
            pass

        # --- Error Level Analysis (ELA) ---
        try:
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Save at quality 95 and compare
            buf = io.BytesIO()
            pil_img.save(buf, "JPEG", quality=95)
            buf.seek(0)
            recompressed = Image.open(buf)

            import numpy as np
            orig_arr = np.array(pil_img, dtype=np.float32)
            recomp_arr = np.array(recompressed, dtype=np.float32)
            diff = np.abs(orig_arr - recomp_arr)
            mean_diff = float(diff.mean())
            max_diff = float(diff.max())
            std_diff = float(diff.std())

            # High variance suggests localized editing
            if std_diff > 15:
                flags.append(f"ELA: High variance ({std_diff:.1f}) — possible localized edits")
                score += 0.25
            elif std_diff > 8:
                flags.append(f"ELA: Moderate variance ({std_diff:.1f})")
                score += 0.1

            if max_diff > 100:
                flags.append(f"ELA: Extreme pixel difference ({max_diff:.0f})")
                score += 0.15
        except ImportError:
            pass  # numpy not available
        except Exception:
            pass

        # --- Image quality consistency ---
        try:
            w, h = pil_img.size
            if w > 100 and h > 100:
                import numpy as np
                arr = np.array(pil_img.convert("L"), dtype=np.float32)
                # Check sharpness consistency across quadrants
                mid_h, mid_w = h // 2, w // 2
                quadrants = [
                    arr[:mid_h, :mid_w], arr[:mid_h, mid_w:],
                    arr[mid_h:, :mid_w], arr[mid_h:, mid_w:],
                ]
                sharpness = []
                for q in quadrants:
                    # Laplacian variance as sharpness proxy
                    lap = np.abs(q[1:-1, 1:-1] * 4
                                 - q[:-2, 1:-1] - q[2:, 1:-1]
                                 - q[1:-1, :-2] - q[1:-1, 2:])
                    sharpness.append(float(lap.var()))
                if max(sharpness) > 0:
                    ratio = min(sharpness) / max(sharpness)
                    if ratio < 0.3:
                        flags.append(f"Quality inconsistency across image regions (ratio: {ratio:.2f})")
                        score += 0.15
        except ImportError:
            pass
        except Exception:
            pass

        score = min(score, 1.0)
        return {
            "manipulation_score": round(score, 3),
            "flags": flags,
            "metadata_anomalies": metadata_anomalies,
        }

    def reverse_image_search(self, image: Union[str, bytes]) -> Dict:
        """Search for the image on the web via Google Cloud Vision.

        Returns:
            Dict with ``matches``, ``sources``, ``earliest_date``,
            ``is_reused``.
        """
        if not self._google_api_key:
            return {"matches": [], "sources": [], "earliest_date": None,
                    "is_reused": False, "error": "No API key provided"}

        img_bytes = self._to_bytes(image)
        try:
            from googleapiclient.discovery import build
            import base64

            service = build("vision", "v1",
                            developerKey=self._google_api_key,
                            cache_discovery=False)

            body = {
                "requests": [{
                    "image": {"content": base64.b64encode(img_bytes).decode("utf-8")},
                    "features": [{"type": "WEB_DETECTION", "maxResults": 10}],
                }]
            }
            response = service.images().annotate(body=body).execute()
            web = response.get("responses", [{}])[0].get("webDetection", {})

            matches = []
            sources = []
            for page in web.get("pagesWithMatchingImages", []):
                url = page.get("url", "")
                title = page.get("pageTitle", "")
                matches.append({"url": url, "title": title})
                sources.append(url)

            return {
                "matches": matches[:10],
                "sources": sources[:10],
                "earliest_date": None,
                "is_reused": len(matches) > 3,
            }
        except Exception as e:
            return {"matches": [], "sources": [], "earliest_date": None,
                    "is_reused": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_ocr_reader(self):
        if self._ocr_reader is not None:
            return self._ocr_reader
        try:
            import easyocr
            self._ocr_reader = easyocr.Reader(self._OCR_LANGS, gpu=False, verbose=False)
            return self._ocr_reader
        except ImportError:
            return None
        except Exception:
            return None

    @staticmethod
    def _to_bytes(image: Union[str, bytes]) -> bytes:
        if isinstance(image, bytes):
            return image
        with open(image, "rb") as f:
            return f.read()
