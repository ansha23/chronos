import glob
import os
from modules.logger import logger

def cleanup_files(config):
    if not config.has_section("file_cleanup"):
        logger.info("No [file_cleanup] section found. Skipping file deletion.")
        return

    patterns = config.get("file_cleanup", "delete_files", fallback="").split(",")
    patterns = [p.strip() for p in patterns if p.strip()]

    if not patterns:
        logger.info("No file patterns specified in delete_files. Nothing to delete.")
        return

    total_deleted = 0
    for pattern in patterns:
        matched_files = glob.glob(pattern)
        for f in matched_files:
            try:
                os.remove(f)
                logger.info(f"Deleted: {f}")
                total_deleted += 1
            except Exception as e:
                logger.warning(f"⚠️ Could not delete {f}: {e}")

    logger.info(f"✅ Cleanup complete. Total files deleted: {total_deleted}")

