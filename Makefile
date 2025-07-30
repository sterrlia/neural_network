.PHONY: build

BUILDDIR = "build"

setup:
	@mkdir build
	@meson setup $(BUILDDIR)

build:
	@meson compile -C $(BUILDDIR)

test:
	@./build/neural_network
