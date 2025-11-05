#!/usr/bin/env python3
"""
Async example of using Synthefy client to make forecast requests.

This example demonstrates how to use the SynthefyAsyncAPIClient to make async
forecast requests, including both basic forecast() and DataFrame-based forecast_dfs()
methods. It also shows concurrent forecasting for multiple scenarios.
"""

import asyncio
import os
import sys
import traceback
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from synthefy import SynthefyAsyncAPIClient
from synthefy.api_client import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from synthefy.data_models import ForecastV2Request, SingleEvalSamplePayload


def create_sample_data():
    """Create sample historical and target data matching the basic example."""

    # Historical data (5 days)
    history_dates = pd.date_range("2023-01-01", periods=5, freq="D")
    history_data = {
        "date": history_dates,
        "sales": [100.0, 110.0, 120.0, 130.0, 140.0],
    }
    history_df = pd.DataFrame(history_data)

    # Target data (3 days to forecast)
    target_dates = pd.date_range("2023-01-06", periods=3, freq="D")
    target_data = {
        "date": target_dates,
        "sales": [np.nan, np.nan, np.nan],  # Values to forecast
    }
    target_df = pd.DataFrame(target_data)

    return history_df, target_df


async def basic_async_forecast():
    """Demonstrate basic async forecast using ForecastV2Request."""

    print("=== Basic Async Forecast ===")

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

    print("Making async forecast request to localhost:8014...")
    print(f"Model: {request.model}")
    print(f"Number of samples: {len(request.samples)}")
    print(
        f"Sample details: {len(request.samples[0])} time series in first scenario"
    )

    try:
        # Configure async client for localhost
        async with SynthefyAsyncAPIClient(
            api_key=None, base_url="http://localhost:8014"
        ) as client:
            print("\nSending async request...")
            response = await client.forecast(request)

            print("\n✅ Async forecast successful!")
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


async def async_dataframe_forecast():
    """Demonstrate async forecast using DataFrames."""

    print("\n=== Async DataFrame Forecast ===")

    # Create sample data
    print("Creating sample data...")
    history_df, target_df = create_sample_data()

    print(f"Historical data shape: {history_df.shape}")
    print(f"Target data shape: {target_df.shape}")
    print(
        f"Historical date range: {history_df['date'].min()} to {history_df['date'].max()}"
    )
    print(
        f"Target date range: {target_df['date'].min()} to {target_df['date'].max()}"
    )

    print("\nHistorical data:")
    print(history_df)
    print("\nTarget data:")
    print(target_df)

    try:
        # Configure async client for localhost
        async with SynthefyAsyncAPIClient(
            api_key=None, base_url="http://localhost:8014"
        ) as client:
            print("\nMaking async forecast request using DataFrames...")

            # Use the async convenience method
            forecast_dfs = await client.forecast_dfs(
                history_dfs=[history_df],
                target_dfs=[target_df],
                target_col="sales",
                timestamp_col="date",
                metadata_cols=[],  # No metadata columns in this example
                leak_cols=[],  # No leak columns in this example
                model="sfm-moe-v1",
            )

            print("\n✅ Async DataFrame forecast successful!")
            print(f"Number of forecast DataFrames: {len(forecast_dfs)}")

            # Display results
            forecast_df = forecast_dfs[0]
            print(f"\nForecast DataFrame shape: {forecast_df.shape}")
            print("\nForecast results:")
            print(forecast_df)

    except BadRequestError as e:
        print(f"❌ Bad Request Error: {e}")
        print(f"Status Code: {e.status_code}")
        if e.response_body:
            print(f"Response Body: {e.response_body}")
    except AuthenticationError as e:
        print(f"❌ Authentication Error: {e}")
        print("Please check your API key")
    except RateLimitError as e:
        print(f"❌ Rate Limit Error: {e}")
        print("Please wait before making another request")
    except APITimeoutError as e:
        print(f"❌ Timeout Error: {e}")
        print("The request took too long to complete")
    except APIConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print(
            "Please check your network connection and ensure the server is running"
        )
    except InternalServerError as e:
        print(f"❌ Server Error: {e}")
        print("The server encountered an internal error")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print(f"Error type: {type(e).__name__}")


async def concurrent_forecasts():
    """Demonstrate concurrent forecasting for multiple scenarios."""

    print("\n=== Concurrent Async Forecasts ===")

    # Create multiple scenarios
    scenarios_history = []
    scenarios_target = []

    for store_id in [1, 2, 3]:
        # Historical data for each store
        hist_dates = pd.date_range("2023-01-01", periods=5, freq="D")
        hist_data = {
            "date": hist_dates,
            "sales": [
                100.0 + store_id * 10,
                110.0 + store_id * 10,
                120.0 + store_id * 10,
                130.0 + store_id * 10,
                140.0 + store_id * 10,
            ],
            "store_id": store_id,
        }
        scenarios_history.append(pd.DataFrame(hist_data))

        # Target data for each store
        target_dates = pd.date_range("2023-01-06", periods=3, freq="D")
        target_data = {
            "date": target_dates,
            "sales": [np.nan, np.nan, np.nan],
            "store_id": store_id,
        }
        scenarios_target.append(pd.DataFrame(target_data))

    print(
        f"Created {len(scenarios_history)} scenarios for concurrent forecasting"
    )

    try:
        async with SynthefyAsyncAPIClient(
            api_key=None, base_url="http://localhost:8014"
        ) as client:
            print("\nMaking concurrent async forecast requests...")

            # Create tasks for concurrent execution
            tasks = []
            for i in range(len(scenarios_history)):
                task = client.forecast_dfs(
                    history_dfs=[scenarios_history[i]],
                    target_dfs=[scenarios_target[i]],
                    target_col="sales",
                    timestamp_col="date",
                    metadata_cols=["store_id"],
                    leak_cols=[],
                    model="sfm-moe-v1",
                )
                tasks.append(task)

            # Execute all forecasts concurrently
            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()

            print(f"\n✅ All {len(results)} concurrent forecasts completed!")
            print(f"Total time: {end_time - start_time:.2f} seconds")

    except BadRequestError as e:
        print(f"❌ Bad Request Error: {e}")
        print(f"Status Code: {e.status_code}")
        if e.response_body:
            print(f"Response Body: {e.response_body}")
    except AuthenticationError as e:
        print(f"❌ Authentication Error: {e}")
        print("Please check your API key")
    except RateLimitError as e:
        print(f"❌ Rate Limit Error: {e}")
        print("Please wait before making another request")
    except APITimeoutError as e:
        print(f"❌ Timeout Error: {e}")
        print("The request took too long to complete")
    except APIConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print(
            "Please check your network connection and ensure the server is running"
        )
    except InternalServerError as e:
        print(f"❌ Server Error: {e}")
        print("The server encountered an internal error")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print(f"Error type: {type(e).__name__}")


async def main():
    """Main function to demonstrate async forecast capabilities."""

    print("Synthefy Async API Client Examples")
    print("=" * 50)

    # Run basic async forecast
    await basic_async_forecast()

    # Run async DataFrame forecast
    await async_dataframe_forecast()

    # Run concurrent forecasts
    await concurrent_forecasts()

    print("\n" + "=" * 50)
    print("All async examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
