import os
import sys
# Ensure project root is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pytest
from types import SimpleNamespace
import app


@pytest.fixture
def client():
    app.app.config['TESTING'] = True
    with app.app.test_client() as client:
        yield client


def test_health(client):
    rv = client.get('/health')
    assert rv.status_code == 200
    assert rv.get_json() == {'status': 'healthy'}


def test_analysis_missing_fields(client):
    # No data at all
    rv = client.post('/analysis', data=json.dumps({}), content_type='application/json')
    assert rv.status_code == 400
    assert 'error' in rv.get_json()

    # Missing sheet_name
    rv = client.post('/analysis', json={'spreadsheet_id': 'id'})
    assert rv.status_code == 400
    assert 'error' in rv.get_json()

    # Missing spreadsheet_id
    rv = client.post('/analysis', json={'sheet_name': 'name'})
    assert rv.status_code == 400
    assert 'error' in rv.get_json()


def test_analysis_success_with_description(client, monkeypatch):
    calls = []

    def fake_initialize_system():
        stub_memory = SimpleNamespace(store=lambda key, value: calls.append((key, value)))
        stub_master = SimpleNamespace(memory=stub_memory)
        return {'master_planner': stub_master}

    def fake_run_sample_workflow(master_planner, initial_data):
        # Check that initial_data is passed correctly
        assert initial_data == {'spreadsheet_id': 'my_id', 'sheet_name': 'my_sheet'}
        return {'status': 'completed', 'results': {'insights': ['ins1', 'ins2']}}

    monkeypatch.setattr(app, 'initialize_system', fake_initialize_system)
    monkeypatch.setattr(app, 'run_sample_workflow', fake_run_sample_workflow)

    rv = client.post('/analysis', json={
        'spreadsheet_id': 'my_id',
        'sheet_name': 'my_sheet',
        'description': 'my_desc'
    })
    assert rv.status_code == 200
    data = rv.get_json()
    assert data['status'] == 'completed'
    assert 'ins1' in data['results']['insights']
    # Ensure description is stored
    assert ('data_description', 'my_desc') in calls


def test_analysis_success_without_description(client, monkeypatch):
    calls = []

    def fake_initialize_system():
        stub_memory = SimpleNamespace(store=lambda key, value: calls.append((key, value)))
        stub_master = SimpleNamespace(memory=stub_memory)
        return {'master_planner': stub_master}

    def fake_run_sample_workflow(master_planner, initial_data):
        assert initial_data == {'spreadsheet_id': 'x', 'sheet_name': 'y'}
        return {'status': 'completed', 'results': {}}

    monkeypatch.setattr(app, 'initialize_system', fake_initialize_system)
    monkeypatch.setattr(app, 'run_sample_workflow', fake_run_sample_workflow)

    rv = client.post('/analysis', json={
        'spreadsheet_id': 'x',
        'sheet_name': 'y'
    })
    assert rv.status_code == 200
    data = rv.get_json()
    assert data == {'status': 'completed', 'results': {}}
    # No description stored
    assert calls == []


def test_analysis_camelcase_fields(client, monkeypatch):
    calls = []

    def fake_initialize_system():
        stub_memory = SimpleNamespace(store=lambda key, value: calls.append((key, value)))
        stub_master = SimpleNamespace(memory=stub_memory)
        return {'master_planner': stub_master}

    def fake_run_sample_workflow(master_planner, initial_data):
        assert initial_data == {'spreadsheet_id': 'id', 'sheet_name': 'name'}
        return {'status': 'ok'}

    monkeypatch.setattr(app, 'initialize_system', fake_initialize_system)
    monkeypatch.setattr(app, 'run_sample_workflow', fake_run_sample_workflow)

    rv = client.post('/analysis', json={'spreadsheetId': 'id', 'sheetName': 'name'})
    assert rv.status_code == 200
    assert rv.get_json() == {'status': 'ok'}