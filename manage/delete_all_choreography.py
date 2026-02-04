#!/usr/bin/env python3
"""
Delete all choreography (choreography field) from all Track entities in Base44.
"""
import os
from manage.base44_utils import get_all_tracks, make_api_request

APP_ID = os.getenv("BASE44_APP_ID", "69668795c37a96600dabcc5c")
ENTITY_TYPE = "Track"

def main():
    print("Fetching all tracks from Base44...")
    tracks = get_all_tracks()
    print(f"Found {len(tracks)} tracks.")
    for t in tracks:
        entity_id = t.get("id") or t.get("_id")
        if not entity_id:
            continue
        if "choreography" in t and t["choreography"]:
            print(f"Deleting choreography for: {t.get('title', 'N/A')} ({entity_id})")
            update_data = {"choreography": []}
            try:
                make_api_request(
                    f"apps/{APP_ID}/entities/{ENTITY_TYPE}/{entity_id}",
                    method="PUT",
                    data=update_data,
                )
            except Exception as e:
                print(f"  Error updating {entity_id}: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
