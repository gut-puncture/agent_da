#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask application for deploying the agent-based data analysis system.
"""

import os
from flask import Flask, request, jsonify

from main import initialize_system, run_sample_workflow

app = Flask(__name__)

@app.route('/analysis', methods=['POST'])
def analyze():
    data = request.get_json() or {}
    spreadsheet_id = data.get('spreadsheet_id') or data.get('spreadsheetId')
    sheet_name = data.get('sheet_name') or data.get('sheetName')
    description = data.get('description')
    if not spreadsheet_id or not sheet_name:
        return jsonify({'error': 'Spreadsheet ID and sheet name are required'}), 400
    try:
        system = initialize_system()
        master_planner = system['master_planner']
        if description:
            master_planner.memory.store('data_description', description)
        initial_data = {
            'spreadsheet_id': spreadsheet_id,
            'sheet_name': sheet_name
        }
        results = run_sample_workflow(master_planner, initial_data)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': 'Analysis failed', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port)