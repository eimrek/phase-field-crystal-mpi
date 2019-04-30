# GNU Makefile

# The compiler
CXX = mpic++

# Compiler parameters
# -Wall		shows all warnings when compiling
# -std=c++11	enables the C++11 standard
# -O3		optimization
CXXFLAGS = -Wall -std=c++11 -O3

# Linker parameters
LFLAGS = -lfftw3_mpi -lfftw3 -lm -lmpi

# Paths
BIN_PATH = bin
OBJ_PATH = obj
OUTPUT_PATH = output

# Name of the applications
APP = $(BIN_PATH)/pfc

# Application compilation settings
APP_CXXFLAGS = $(CXXFLAGS) -Iinclude

# Object files
OBJS = obj/main.o obj/pfc.o obj/mechanical_equilibrium.o

####################
# MAIN APP TARGETS #
####################

# Default target to run all targets to build the application
all: $(APP)

# Target that links the objects to the executable
# NB: objects are a dependency
$(APP): $(OBJS)
	@mkdir -p $(BIN_PATH)
	$(CXX) $(OBJS) -o $(APP) $(LFLAGS)

# Target that compiles sources to object files
obj/%.o: src/%.cpp
	@mkdir -p $(OBJ_PATH)
	$(CXX) $(APP_CXXFLAGS) -c $< -o $@

run:
	@mkdir -p $(OUTPUT_PATH)
	$(MPI_LOC)/bin/mpirun -n 4 $(APP)

#################
# OTHER TARGETS #
#################

doc:
	doxygen

clean:
	rm -rf $(OBJ_PATH)
	rm -rf $(BIN_PATH)
	rm -rf $(OUTPUT_PATH)
	rm -rf docs/html
