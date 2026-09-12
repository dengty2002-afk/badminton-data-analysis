CREATE TABLE `calibrations` (
	`id` text PRIMARY KEY NOT NULL,
	`video_id` text NOT NULL,
	`version` integer NOT NULL,
	`frame_time_ms` integer NOT NULL,
	`image_width` integer NOT NULL,
	`image_height` integer NOT NULL,
	`corners_json` text NOT NULL,
	`homography_json` text NOT NULL,
	`court_orientation` text DEFAULT 'upper_is_far_side' NOT NULL,
	`quality_status` text DEFAULT 'accepted' NOT NULL,
	`source` text DEFAULT 'manual' NOT NULL,
	`calibrator_id` text DEFAULT 'private-user' NOT NULL,
	`created_at` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `idx_calibrations_video_version` ON `calibrations` (`video_id`,`version`);