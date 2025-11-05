#!/usr/bin/env python3
"""
Basic example of using Synthefy client to make forecast requests.

This example demonstrates how to use the SynthefyAPIClient to make a forecast request
that matches the curl command:
curl -X POST "http://localhost:8014/api/v2/foundation_models/forecast/stream" \
  -H "Content-Type: application/json" \
  -d '{ "model": "sfm-moe-v1", "samples": [ [ { "sample_id": "sales", "history_timestamps": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"], "history_values": [100.0, 110.0, 120.0, 130.0, 140.0], "target_timestamps": ["2023-01-06", "2023-01-07", "2023-01-08"], "target_values": [null, null, null], "forecast": true, "metadata": false, "leak_target": false, "column_name": "sales" } ] ] }'
"""

import os
import sys
import traceback
from typing import Optional

from synthefy import SynthefyAPIClient
from synthefy.api_client import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from synthefy.data_models import ForecastV2Request, SingleEvalSamplePayload


def main():
    """Main function to demonstrate basic forecast request."""

    # Check for API key
    api_key = None

    # Create the forecast request matching the curl example
    samples = [
        [
            SingleEvalSamplePayload(
                sample_id="sales",
                history_timestamps=[
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                ],
                history_values=[100.0, 110.0, 120.0, 130.0, 140.0],
                target_timestamps=["2023-01-06", "2023-01-07", "2023-01-08"],
                target_values=[None, None, None],
                forecast=True,
                metadata=False,
                leak_target=False,
                column_name="sales",
            )
        ]
    ]

    request = ForecastV2Request(samples=samples, model="sfm-moe-v1")

    print("Making forecast request to localhost:8014...")
    print(f"Model: {request.model}")
    print(f"Number of samples: {len(request.samples)}")
    print(
        f"Sample details: {len(request.samples[0])} time series in first scenario"
    )

    try:
        # Configure client for localhost
        with SynthefyAPIClient(
            api_key=api_key, base_url="http://localhost:8014"
        ) as client:
            print("\nSending request...")
            response = client.forecast(request)

            print("\n✅ Forecast successful!")
            print(f"Number of forecast scenarios: {len(response.forecasts)}")

            # Display results
            for i, forecast_scenario in enumerate(response.forecasts):
                print(f"\nScenario {i + 1}:")
                for forecast in forecast_scenario:
                    print(f"  Sample ID: {forecast.sample_id}")
                    print(f"  Model: {forecast.model_name}")
                    print(f"  Timestamps: {forecast.timestamps}")
                    print(f"  Values: {forecast.values}")

    except BadRequestError as e:
        print(f"❌ Bad Request Error: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        print(f"Status Code: {e.status_code}")
        if e.response_body:
            print(f"Response Body: {e.response_body}")
    except AuthenticationError as e:
        print(f"❌ Authentication Error: {e}")
        print("Please check your API key")
        print(f"Stack trace: {traceback.format_exc()}")
    except RateLimitError as e:
        print(f"❌ Rate Limit Error: {e}")
        print("Please wait before making another request")
        print(f"Stack trace: {traceback.format_exc()}")
    except APITimeoutError as e:
        print(f"❌ Timeout Error: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        print("The request took too long to complete")
    except APIConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print(
            "Please check your network connection and ensure the server is running"
        )
        print(f"Stack trace: {traceback.format_exc()}")
    except InternalServerError as e:
        print(f"❌ Server Error: {e}")
        print("The server encountered an internal error")
        print(f"Stack trace: {traceback.format_exc()}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Stack trace: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
