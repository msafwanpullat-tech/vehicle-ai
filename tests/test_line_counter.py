import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.services.counter import LineCounter

def test_line_crossing_math():
    print("Testing LineCounter line crossing logic & duplicate suppression...")
    
    counter = LineCounter(line_position_ratio=0.5)
    counter.configure_frame(width=1000, height=1000, line_ratio=0.5) # Line at Y=500

    # Vehicle 1: Moves downwards (Y=450 -> Y=550) -> Should trigger IN count
    evt1 = counter.update_track(track_id=1, class_name="car", center=(500, 450))
    assert evt1 is None, "Should not trigger crossing before reaching line"

    evt2 = counter.update_track(track_id=1, class_name="car", center=(500, 550))
    assert evt2 is not None, "Vehicle 1 should trigger crossing event"
    assert evt2["direction"] == "IN", "Vehicle 1 direction should be IN"
    assert evt2["total_count"] == 1, "Total count should be 1"

    # Vehicle 1: Moves further down (Y=550 -> Y=600) -> Should NOT count again
    evt3 = counter.update_track(track_id=1, class_name="car", center=(500, 600))
    assert evt3 is None, "Vehicle 1 should NOT be counted again!"

    # Vehicle 2: Moves upwards (Y=550 -> Y=450) -> Should trigger OUT count
    counter.update_track(track_id=2, class_name="bus", center=(300, 550))
    evt4 = counter.update_track(track_id=2, class_name="bus", center=(300, 450))
    assert evt4 is not None, "Vehicle 2 should trigger crossing event"
    assert evt4["direction"] == "OUT", "Vehicle 2 direction should be OUT"

    # Vehicle 3: Stays on one side of the line (Y=100 -> Y=200) -> Should NOT count
    counter.update_track(track_id=3, class_name="truck", center=(200, 100))
    evt5 = counter.update_track(track_id=3, class_name="truck", center=(200, 200))
    assert evt5 is None, "Vehicle 3 should NOT count as it never crosses Y=500"

    summary = counter.get_summary()
    print("\nLineCounter Test Summary:", summary)
    
    assert summary["total_line_crossings"] == 2, f"Expected 2 total crossings, got {summary['total_line_crossings']}"
    assert summary["in_count"] == 1, f"Expected 1 IN count, got {summary['in_count']}"
    assert summary["out_count"] == 1, f"Expected 1 OUT count, got {summary['out_count']}"
    assert summary["class_line_counts"]["car"] == 1, "Expected 1 car count"
    assert summary["class_line_counts"]["bus"] == 1, "Expected 1 bus count"
    assert summary["class_line_counts"]["truck"] == 0, "Expected 0 truck count"

    print("\n[SUCCESS] All LineCounter mathematical and logic tests passed!")

if __name__ == "__main__":
    test_line_crossing_math()
