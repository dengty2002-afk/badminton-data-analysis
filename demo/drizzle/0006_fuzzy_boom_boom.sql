ALTER TABLE `semantic_gold_revisions` ADD `landing_status` text DEFAULT 'observed' NOT NULL;--> statement-breakpoint
ALTER TABLE `semantic_gold_revisions` ADD `landing_area` integer;--> statement-breakpoint
ALTER TABLE `semantic_gold_revisions` ADD `landing_area_schema` text DEFAULT 'shuttlelab_9x7_v1' NOT NULL;--> statement-breakpoint
PRAGMA optimize;
