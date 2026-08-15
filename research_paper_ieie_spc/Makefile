.PHONY: all figures validate-public validate-private release paper check clean

all: figures validate-public paper check

figures:
	MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/ieie-mpl XDG_CACHE_HOME=/private/tmp/ieie-cache python3 scripts/generate_figures.py

validate-public:
	python3 ../tools/test_result_analysis.py
	python3 ../tools/test_followup_plan.py
	python3 ../tools/validate_suite.py
	python3 ../tools/verify_colab_parity.py

validate-private:
	python3 scripts/validate_followup_predictions.py

release:
	python3 ../tools/build_public_release.py

paper:
	tectonic main.tex --keep-logs --keep-intermediates

check:
	python3 scripts/check_submission.py

clean:
	rm -f main.aux main.log main.out main.xdv
