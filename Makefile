# SYCL Particle Simulation Makefile
# ================================

# Compiler detection
CXX := $(shell which icpx dpcpp clang++ | head -1)
ifeq ($(CXX),)
    $(error No SYCL compiler found. Install Intel oneAPI or DPC++)
endif

# Directories
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
TARGET := $(BUILD_DIR)/particle_simulation

# Compiler flags
PYTHON_INCLUDE := $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB := $(shell python3 -c "import sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), 'libpython3.12.so'))")
CXXFLAGS := -fsycl -I$(INC_DIR) -I$(PYTHON_INCLUDE) -O3 -std=c++17 -DMKL_ILP64 -qmkl
LDFLAGS := $(PYTHON_LIB)

# Source files
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Default target
.PHONY: all clean help run-example

all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o $@
	@echo "Build successful! Executable: $(TARGET)"

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
	rm -f *.out
	@echo "Clean completed!"

# Example run (requires input files)
run-example: $(TARGET)
	@echo "Example usage (replace with actual input files):"
	@echo "$(TARGET) input1.bin input2.bin gpu 256 24"

# Help target
help:
	@echo "SYCL Particle Simulation Build System"
	@echo "====================================="
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build the simulation (default)"
	@echo "  clean       - Remove build files and output"
	@echo "  help        - Show this help message"
	@echo "  run-example - Show example usage"
	@echo ""
	@echo "Usage:"
	@echo "  make                    # Build simulation"
	@echo "  make clean             # Clean build files"
	@echo "  ./build/particle_simulation input1.bin input2.bin gpu 256 24"
	@echo ""
	@echo "Or use the Python runner:"
	@echo "  python scripts/run.py input1.bin input2.bin --device gpu --wg-size 256 --prob-size 24"

# Dependency tracking
-include $(OBJECTS:.o=.d)
$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@$(CXX) $(CXXFLAGS) -MM -MT $(@:.d=.o) $< > $@