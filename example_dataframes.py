#!/usr/bin/env python3
"""
Example using pandas DataFrames with Synthefy client.

This example demonstrates how to use the convenience forecast_dfs() method
with pandas DataFrames for the same forecast scenario as the basic example.
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from synthefy import SynthefyAPIClient
from synthefy.api_client import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)


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


def main():
    """Main function to demonstrate DataFrame-based forecast request."""

    # Check for API key
    api_key = None

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
        # Configure client for localhost
        with SynthefyAPIClient(
            api_key=api_key, base_url="http://localhost:8014"
        ) as client:
            print("\nMaking forecast request using DataFrames...")

            # Use the convenience method
            forecast_dfs = client.forecast_dfs(
                history_dfs=[history_df],
                target_dfs=[target_df],
                target_col="sales",
                timestamp_col="date",
                metadata_cols=[],  # No metadata columns in this example
                leak_cols=[],  # No leak columns in this example
                model="sfm-moe-v1",
            )

            print("\n✅ Forecast successful!")
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


if __name__ == "__main__":
    main()
