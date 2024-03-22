 
 .PHONY: format
 format:
	black --config pyproject.toml .
	isort --profile black .
