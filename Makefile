# SoundKey — convenience targets. Everything here is pure Python standard
# library; nothing to install. Author: Krishita Sanjay Choksi.

PY ?= python3

.PHONY: all test header figures eval demos gif media clean sim-help

all: header figures eval media  ## Regenerate every generated artifact + run checks
	$(PY) -m unittest discover -s tests

test:  ## Run the test suite
	$(PY) -m unittest discover -s tests -v

header:  ## Regenerate the on-device C cue header from the cue library
	$(PY) cues/generate_c_header.py

figures:  ## Regenerate architecture, flow, and table figures
	$(PY) figures/generate_figures.py

eval:  ## Run the evaluation harness and regenerate its charts
	$(PY) eval/run_eval.py

demos:  ## Render audio demos (WAV + transcripts) into media/
	cd sim && $(PY) -m soundkey_sim.cli export-demos ../media

gif:  ## Render the demo GIF
	$(PY) media/generate_gif.py

media: demos gif  ## All audio + GIF demos

sim-help:  ## Show the simulator CLI help
	cd sim && $(PY) -m soundkey_sim.cli --help

clean:  ## Remove Python caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
