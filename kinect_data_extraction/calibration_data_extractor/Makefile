INPUT_FILE := ../../samples/sample.mkv
OUTPUT_DIR := ./outputs
OUTPUT_FILE := ${OUTPUT_DIR}/camera_calibration_params.json

.PHONY: download
download:
	$(MAKE) -C ../../samples download

.PHONY: run
run:
	-rm ${OUTPUT_FILE}
	mkdir -p ${OUTPUT_DIR}
	./build/k4a_calibration_data_extractor ${INPUT_FILE} ${OUTPUT_FILE}

.PHONY: clean
clean:
	rm -rf ./build
	rm -rf ${OUTPUT_DIR}
