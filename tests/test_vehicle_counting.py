import unittest

from count_vehicles import InvalidVideoError, VehicleCrossingCounter, validate_decoded_frame_count


class VehicleCrossingCounterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.class_counts = {"car": 0, "bus": 0}
        self.counter = VehicleCrossingCounter(self.class_counts)

    def test_counts_a_track_once_for_crossings_in_either_direction(self) -> None:
        self.counter.observe(1, "car", (20, 90), 100)
        class_name, direction, counted = self.counter.observe(1, "car", (20, 110), 100)
        self.assertEqual((class_name, direction, counted), ("car", "down", True))

        class_name, direction, counted = self.counter.observe(1, "car", (20, 90), 100)
        self.assertEqual((class_name, direction, counted), ("car", "up", False))
        self.assertEqual(self.counter.total_count, 1)

    def test_class_counts_only_change_on_crossing_and_total_matches_classes(self) -> None:
        self.counter.observe(1, "car", (20, 80), 100)
        self.counter.observe(1, "car", (20, 90), 100)
        self.assertEqual(self.counter.total_count, 0)

        self.counter.observe(1, "car", (20, 110), 100)
        self.counter.observe(2, "bus", (40, 90), 100)
        self.counter.observe(2, "bus", (40, 110), 100)

        self.assertEqual(self.class_counts, {"car": 1, "bus": 1})
        self.assertEqual(self.counter.total_count, sum(self.class_counts.values()))

    def test_new_track_id_can_count_the_same_physical_vehicle_again(self) -> None:
        self.counter.observe(7, "car", (20, 90), 100)
        self.counter.observe(7, "car", (20, 110), 100)

        self.counter.observe(23, "car", (20, 90), 100)
        self.counter.observe(23, "car", (20, 110), 100)

        self.assertEqual(self.class_counts["car"], 2)
        self.assertEqual(self.counter.total_count, 2)

    def test_detections_without_track_ids_are_not_counted(self) -> None:
        self.counter.observe(None, "car", (20, 90), 100)
        self.counter.observe(None, "car", (20, 110), 100)

        self.assertEqual(self.counter.total_count, 0)
        self.assertEqual(sum(self.class_counts.values()), 0)


class DecodedFrameValidationTests(unittest.TestCase):
    def test_accepts_all_decoded_frames_when_metadata_overstates_fragmented_mp4(self) -> None:
        validate_decoded_frame_count(expected_frames=225, frames_processed=223)

    def test_rejects_files_without_multiple_decodable_frames(self) -> None:
        for frame_count in (0, 1):
            with self.subTest(frame_count=frame_count):
                with self.assertRaisesRegex(InvalidVideoError, "unsupported codec"):
                    validate_decoded_frame_count(
                        expected_frames=225, frames_processed=frame_count
                    )


if __name__ == "__main__":
    unittest.main()
