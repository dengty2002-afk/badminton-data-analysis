CREATE TABLE `semantic_gold_revisions` (
	`id` text PRIMARY KEY NOT NULL,
	`video_id` text NOT NULL,
	`anchor_id` text NOT NULL,
	`revision` integer NOT NULL,
	`rally_id` text NOT NULL,
	`hit_frame` integer NOT NULL,
	`hitter` text NOT NULL,
	`stroke_type` text DEFAULT '' NOT NULL,
	`landing_x` real,
	`landing_y` real,
	`landing_frame` integer,
	`landing_kind` text,
	`confidence` text DEFAULT 'high' NOT NULL,
	`uncertainty_frames` integer DEFAULT 0 NOT NULL,
	`notes` text DEFAULT '' NOT NULL,
	`status` text NOT NULL,
	`annotator_id` text DEFAULT 'public-annotator' NOT NULL,
	`created_at` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `idx_semantic_gold_video_anchor_revision` ON `semantic_gold_revisions` (`video_id`,`anchor_id`,`revision`);--> statement-breakpoint
CREATE INDEX `idx_semantic_gold_video_status` ON `semantic_gold_revisions` (`video_id`,`status`);--> statement-breakpoint
PRAGMA optimize;
