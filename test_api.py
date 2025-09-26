#!/usr/bin/env python3
"""
Test script for the Crop Yield Prediction API
"""
import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_prediction():
    """Test the prediction endpoint"""
    print("Testing prediction endpoint...")
    
    # Sample data matching the schema (from CSV image - second row)
    payload = {
        "Crop": "Arhar/Tur",
        "Crop_Year": 1997,
        "Season": "Kharif     ",
        "State": "Assam",
        "Area": 6637.0,
        "Annual_Rainfall": 2051.4,
        "Fertilizer": 631643.29,
        "Pesticide": 2057.47
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Yield: {result['predicted_yield']:.4f} {result['unit']}")
            print(f"Recommendations: {result['recommendations']}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
    except Exception as e:
        print(f"Error: {e}")
    
    print()

def test_coconut_rejection():
    """Test that Coconut crops are rejected"""
    print("Testing Coconut crop rejection...")
    
    payload = {
        "Crop": "Coconut ",
        "Crop_Year": 2020,
        "Season": "Whole Year ",
        "State": "Kerala",
        "Area": 50.0,
        "Annual_Rainfall": 2500.0,
        "Fertilizer": 2000.0,
        "Pesticide": 10.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print()

if __name__ == "__main__":
    print("=== Crop Yield Prediction API Test ===\n")
    
    test_health()
    test_prediction()
    test_coconut_rejection()
    
    print("=== Test Complete ===")
