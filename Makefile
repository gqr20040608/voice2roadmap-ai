setup:
	python -m pip install -r requirements.txt
run:
	streamlit run app.py
test:
	python -m pytest -q
