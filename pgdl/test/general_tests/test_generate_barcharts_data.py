import importlib.util
from pathlib import Path


MODULE_PATH = Path(
    "/home/why/dbagent/pgdl/paper_curated_20260622/anchored_generation/main_result_barcharts/generate_barcharts.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("generate_barcharts", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_chart_data_reads_csv_and_computes_sst2_speedup():
    module = load_module()

    chart_data = module.load_chart_data(module.DATA_CSV_PATH)

    assert chart_data["latency"]["sst-2"][module.DBAGENT_IDX] == [
        5.0757,
        6.3184,
        9.9839,
        10.9146,
        11.9554,
    ]
    assert chart_data["throughput"]["sst-2"][module.DBAGENT_IDX] == 224.0
    assert chart_data["max_speedup"]["sst-2"] > 1.15
