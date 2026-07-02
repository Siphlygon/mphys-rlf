LOG_DIRNAME ?= logs
DATASET_CONFIG ?= dataset
GENERATED_CONFIG ?= generated

ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LOG_DIR ?= $(abspath $(dir $(ROOT_DIR)))/$(LOG_DIRNAME)

USE_SLURM = FALSE

SLURM_CONSTRAINT ?= A100
SLURM_TIME ?= 1-23
SLURM_OUTPUT_FORMAT ?= out-slurm_%j.out
SLURM_CPUS_PER_TASK ?= 16
SLURM_NODE_DENYLIST ?= compute-0-9,compute-0-1,compute-0-2
SLURM_OUTPUT := $(LOG_DIR)/$(SLURM_OUTPUT_FORMAT)

SLURM_SAMPLE_ARRAYSIZE ?= 6
SLURM_ANALYSIS_ARRAYSIZE ?= 11

#maxindex variables evaluate in the shell - can't be compared in make directly
SLURM_SAMPLE_ARRAYSIZE_MAXINDEX := $$(( $(SLURM_SAMPLE_ARRAYSIZE) - 1 ))
SLURM_ANALYSIS_ARRAYSIZE_MAXINDEX := $$(( $(SLURM_ANALYSIS_ARRAYSIZE) - 1 ))
SLURM_SAMPLE_ARRAY := 0-$(SLURM_SAMPLE_ARRAYSIZE_MAXINDEX)
SLURM_ANALYSIS_ARRAY := 0-$(SLURM_ANALYSIS_ARRAYSIZE_MAXINDEX)

SLURM_ARGS := --export=ALL,HDF5_USE_FILE_LOCKING=FALSE --wait --constraint=$(SLURM_CONSTRAINT) --time=$(SLURM_TIME) --output=$(SLURM_OUTPUT) --chdir=$(ROOT_DIR) --cpus-per-task=$(SLURM_CPUS_PER_TASK)
ifneq ($(SLURM_NODE_DENYLIST),NONE)
	SLURM_ARGS := $(SLURM_ARGS) --exclude=$(SLURM_NODE_DENYLIST)
endif

SLURM_SAMPLE_ARGS = --array=$(SLURM_SAMPLE_ARRAY)
SLURM_ANALYSIS_ARGS = --array=$(SLURM_ANALYSIS_ARRAY)

define RUN_JOB
	if [ "$(USE_SLURM)" = "TRUE" ]; then \
		sbatch --job-name="$@" $(SLURM_ARGS) $$(SLURM_EXTRA_ARGS) --wrap="$$(PYTHONINSTR)"; \
	else \
		$$PYTHONINSTR; \
	fi
endef

create_folders:
	mkdir -p $(LOG_DIR)
	PYTHONINSTR="python $(ROOT_DIR)/program_prep/create_folders.py --configs $(GENERATED_CONFIG) $(DATASET_CONFIG)"; \
	$(RUN_JOB)


copy_sampling_configs: create_folders
	PYTHONINSTR="python $(ROOT_DIR)/program_prep/copy_configs.py --configs $(GENERATED_CONFIG)"; \
	$(RUN_JOB)

save_maxvals: create_folders
	PYTHONINSTR="python $(ROOT_DIR)/dataset_prep/save_maxvals.py --configs $(GENERATED_CONFIG) $(DATASET_CONFIG)"; \
	$(RUN_JOB)

sample_model: copy_sampling_configs save_maxvals
	PYTHONINSTR="python $(ROOT_DIR)/sampling/generate_fits_files.py --configs $(GENERATED_CONFIG)"; \
	SLURM_EXTRA_ARGS="$(SLURM_SAMPLE_ARGS)" \
	$(RUN_JOB)

run_analysis: sample_model
	PYTHONINSTR="python $(ROOT_DIR)/analysis/run_analysis.py --configs $(GENERATED_CONFIG) $(DATASET_CONFIG)"; \
	SLURM_EXTRA_ARGS="$(SLURM_ANALYSIS_ARGS)" \
	$(RUN_JOB)

plot_histograms: run_analysis
	PYTHONINSTR="python $(ROOT_DIR)/plotting/plot_histograms.py --configs $(GENERATED_CONFIG) $(DATASET_CONFIG)"; \
	$(RUN_JOB)

estimate_completeness: run_analysis
	PYTHONINSTR="python $(ROOT_DIR)/completeness/completeness_estim.py --configs $(GENERATED_CONFIG)"; \
	$(RUN_JOB)

plot_model_vs_peak: run_analysis
	PYTHONINSTR="python $(ROOT_DIR)/plotting/model_vs_peak.py --configs $(GENERATED_CONFIG) $(DATASET_CONFIG)"; \
	$(RUN_JOB)

#insert other plotting here

paper_plots: plot_histograms estimate_completeness plot_model_vs_peak

plot_rlf: estimate_completeness
	PYTHONINSTR="python $(ROOT_DIR)/rlf/rlf.py --configs $(DATASET_CONFIG)"; \
	$(RUN_JOB)

