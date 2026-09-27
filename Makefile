SHELL := /bin/bash
CLUSTER_NAME := nvkind-cluster
DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: all help start stop enter-cp logs clean run-collectives run-sharding run-transformer

help:
	@echo "=========================================================="
	@echo "         JAX Distributed GPU Cluster Lab"
	@echo "=========================================================="
	@echo "Commands:"
	@echo "  make start            - Start KIND cluster with GPU support & mounted workspace"
	@echo "  make enter-cp         - Step into the control-plane container interactive shell"
	@echo "  make stop             - Tear down the KIND cluster"
	@echo "  make run-collectives  - Run 4-rank JAX Collectives on RTX 5090"
	@echo "  make run-sharding     - Run 4-GPU 2D Rack Mesh Sharding MatMul"
	@echo "  make run-transformer  - Run 4-GPU Megatron-style Transformer MLP"
	@echo "  make logs             - View logs from the 4 ranks"
	@echo "  make clean            - Kill any stale python cluster processes"
	@echo "=========================================================="

start:
	@export PATH="$$HOME/.local/bin:$$PATH"; \
	if kind get clusters 2>/dev/null | grep -q "^$(CLUSTER_NAME)$$"; then \
		echo "Cluster $(CLUSTER_NAME) is already running."; \
	else \
		echo "Creating kind cluster $(CLUSTER_NAME)..."; \
		kind create cluster --config $(DIR)/kind-cluster-gpu.yaml; \
	fi
	@echo ""
	@echo "Cluster is ready! You can now step into the control-plane with:"
	@echo "  make enter-cp"

enter-cp:
	@docker exec -it $(CLUSTER_NAME)-control-plane bash -c "cd /workspace && exec bash"

stop:
	@export PATH="$$HOME/.local/bin:$$PATH"; \
	kind delete cluster --name $(CLUSTER_NAME)

run-collectives:
	@$(DIR)/launch_cluster.sh 4 12380 $(DIR)/01_multi_rank_collectives.py

run-sharding:
	@$(DIR)/launch_cluster.sh 4 12381 $(DIR)/02_sharding_and_placement.py

run-transformer:
	@$(DIR)/launch_cluster.sh 4 12382 $(DIR)/03_model_parallel_transformer.py

logs:
	@for r in 0 1 2 3; do \
		echo "==================== [Rank $$r Log] ===================="; \
		cat $(DIR)/rank_$$r.log 2>/dev/null || echo "No log for rank $$r"; \
		echo ""; \
	done

clean:
	@pkill -9 -f "01_multi_rank_collectives.py" 2>/dev/null || true
	@pkill -9 -f "02_sharding_and_placement.py" 2>/dev/null || true
	@pkill -9 -f "03_model_parallel_transformer.py" 2>/dev/null || true
	@echo "Cleaned up active cluster processes."

run-profile:
	@rm -rf $(DIR)/traces
	@$(DIR)/launch_cluster.sh 4 12385 $(DIR)/04_profiled_collectives.py
	@echo ""
	@echo "=========================================================="
	@echo " Trace captured! Files located in $(DIR)/traces:"
	@find $(DIR)/traces -type f
	@echo ""
	@echo " To view: Open https://ui.perfetto.dev and drag in the .trace.json.gz file"
	@echo "=========================================================="
