CREATE TABLE `event_revisions` (
	`id` text PRIMARY KEY NOT NULL,
	`video_id` text NOT NULL,
	`event_id` text NOT NULL,
	`revision` integer NOT NULL,
	`candidate_frame` integer NOT NULL,
	`candidate_time_ms` integer NOT NULL,
	`rally_id` text DEFAULT 'Model pilot' NOT NULL,
	`hitter` text NOT NULL,
	`stroke` text NOT NULL,
	`backhand` integer DEFAULT false NOT NULL,
	`aroundhead` integer DEFAULT false NOT NULL,
	`certainty` text DEFAULT '确定' NOT NULL,
	`notes` text DEFAULT '' NOT NULL,
	`status` text NOT NULL,
	`label_source` text DEFAULT 'machine_review' NOT NULL,
	`machine_prediction_json` text DEFAULT '{}' NOT NULL,
	`annotator_id` text DEFAULT 'public-annotator' NOT NULL,
	`created_at` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `idx_event_revisions_video_event_revision` ON `event_revisions` (`video_id`,`event_id`,`revision`);--> statement-breakpoint
CREATE INDEX `idx_event_revisions_video_status` ON `event_revisions` (`video_id`,`status`);