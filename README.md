# 2019 NG Signal Challenge

Repo for 2019 NG signal classification challenge

# Setup and Run
    # Train
	train.m
	
	# Test
	test.m
	
	# Test single sound file
	identifySound.m

# Structure
  - `data`: `train` and `test` data
  - `scraper`
    - `google_audioset`: Scrapes Youtube for audio samples following AudioSet dataset
    - `macaulay`: Scrapes Macaulay Library for animal sounds
  - `models`: Trained classifiers