.PHONY: build

BUILDDIR = "build"

setup:
	meson setup $(BUILDDIR)

build:
	meson compile -C $(BUILDDIR)

run:
	@./build/neural_network
