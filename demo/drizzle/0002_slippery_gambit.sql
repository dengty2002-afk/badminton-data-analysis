CREATE TABLE `player_side_assignments` (
	`id` text PRIMARY KEY NOT NULL,
	`video_id` text NOT NULL,
	`version` integer NOT NULL,
	`effective_frame` integer NOT NULL,
	`effective_time_ms` integer NOT NULL,
	`upper_player` text NOT NULL,
	`lower_player` text NOT NULL,
	`source` text DEFAULT 'manual_side_switch' NOT NULL,
	`editor_id` text DEFAULT 'public-annotator' NOT NULL,
	`created_at` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `idx_player_sides_video_version` ON `player_side_assignments` (`video_id`,`version`);--> statement-breakpoint
CREATE INDEX `idx_player_sides_video_frame` ON `player_side_assignments` (`video_id`,`effective_frame`);