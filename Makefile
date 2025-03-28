.PHONY: setup test clean verify

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements/dev.txt
	. venv/bin/activate && python scripts/verify_setup.py

test:
	. venv/bin/activate && pytest tests/

verify:
	. venv/bin/activate && python scripts/verify_setup.py

clean:
	rm -rf venv
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete 