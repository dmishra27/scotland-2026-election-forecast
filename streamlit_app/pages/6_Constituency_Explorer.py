"""
Page 6 — Constituency Explorer.

Live search across all 73 constituency seats via the forecast API,
with a styled results table and a vote-share detail card.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")

PARTY_COLOURS = {
    "SNP": "#D4A017",
    "Reform": "#1ABC9C",
    "Labour": "#C0392B",
    "Conservative": "#2471A3",
    "LibDem": "#F39C12",
    "Green": "#27AE60",
}

# Certified Electoral Management Board results — 8 May 2026, all 129 seats declared.
ACTUAL_RESULTS: dict[str, dict] = {
    "Orkney Islands": {
        "actual_winner": "LibDem",
        "actual_majority": 5560,
        "actual_majority_pp": 54.07,
        "turnout_pct": 57.64,
        "actual_shares": {
            "LibDem": 70.22, "SNP": 16.15, "Reform": 8.21,
            "Conservative": 3.48, "Labour": 1.94,
        },
    },
    "Inverclyde": {
        "actual_winner": "SNP",
        "actual_majority": 5317,
        "actual_majority_pp": 16.60,
        "turnout_pct": 51.55,
        "actual_shares": {
            "SNP": 44.32, "Labour": 27.72, "Reform": 17.64,
            "LibDem": 6.10, "Conservative": 4.22,
        },
    },
    "Inverness and Nairn": {
        "actual_winner": "SNP",
        "actual_majority": 427,
        "actual_majority_pp": 1.16,
        "turnout_pct": 54.54,
        "actual_shares": {
            "SNP": 30.39, "LibDem": 29.22, "Independent": 21.34,
            "Reform": 10.32, "Labour": 4.69, "Conservative": 3.74,
        },
    },
    "Strathkelvin and Bearsden": {
        "actual_winner": "LibDem",
        "actual_majority": 2572,
        "actual_majority_pp": 6.47,
        "turnout_pct": 62.72,
        "actual_shares": {
            "LibDem": 39.46, "SNP": 33.00, "Labour": 11.76,
            "Reform": 10.44, "Conservative": 5.33,
        },
    },
    "Edinburgh Central": {
        "actual_winner": "Green",
        "actual_majority": 4582,
        "actual_majority_pp": 13.00,
        "turnout_pct": 54.54,
        "actual_shares": {
            "Green": 35.98, "Labour": 22.98, "SNP": 21.86,
            "Conservative": 6.42, "LibDem": 6.15, "Reform": 5.32,
        },
    },
    "Na h-Eileanan an Iar": {
        "actual_winner": "Labour",
        "actual_majority": 154,
        "actual_majority_pp": 1.25,
        "turnout_pct": 56.70,
        "actual_shares": {
            "Labour": 37.72, "SNP": 36.48, "Reform": 13.14,
            "LibDem": 6.57, "Conservative": 4.80,
        },
    },
    "Glasgow Southside": {
        "actual_winner": "Green",
        "actual_majority": 3101,
        "actual_majority_pp": 8.05,
        "turnout_pct": 59.09,
        "actual_shares": {
            "Green": 36.49, "SNP": 28.43, "Labour": 18.96,
            "Reform": 7.84, "Conservative": 3.59, "LibDem": 2.97,
        },
    },
    "Fife North East": {
        "actual_winner": "LibDem",
        "actual_majority": 13474,
        "actual_majority_pp": 40.22,
        "turnout_pct": 58.42,
        "actual_shares": {
            "LibDem": 63.72, "SNP": 23.51, "Reform": 7.53,
            "Conservative": 2.79, "Labour": 2.44,
        },
    },
    "Edinburgh Southern": {
        "actual_winner": "Labour",
        "actual_majority": 4963,
        "actual_majority_pp": 12.55,
        "turnout_pct": 60.98,
        "actual_shares": {
            "Labour": 42.88, "SNP": 30.33, "Conservative": 8.65,
            "LibDem": 8.43, "Reform": 8.38,
        },
    },
    "Aberdeenshire West": {
        "actual_winner": "Conservative",
        "actual_majority": 5784,
        "actual_majority_pp": 15.62,
        "turnout_pct": 60.47,
        "actual_shares": {
            "Conservative": 42.92, "SNP": 27.30, "Reform": 14.76,
            "LibDem": 10.79, "Labour": 4.24,
        },
    },
    "Dumbarton": {
        "actual_winner": "Labour",
        "actual_majority": 1786,
        "actual_majority_pp": 5.58,
        "turnout_pct": 56.84,
        "actual_shares": {
            "Labour": 39.81, "SNP": 34.23, "Reform": 15.74,
            "Conservative": 4.27, "LibDem": 3.73,
        },
    },
    "Aberdeen Central": {
        "actual_winner": "SNP", "actual_majority": 6972,
        "actual_majority_pp": 25.67, "turnout_pct": 46.88,
        "actual_shares": {"SNP": 44.08, "Labour": 18.41, "Reform": 14.49, "Conservative": 13.58, "LibDem": 9.44},
    },
    "Aberdeen Deeside and North Kincardine": {
        "actual_winner": "SNP", "actual_majority": 1244,
        "actual_majority_pp": 3.60, "turnout_pct": 54.87,
        "actual_shares": {"SNP": 34.11, "Conservative": 30.51, "Reform": 17.69, "LibDem": 8.33, "Labour": 8.12},
    },
    "Aberdeen Donside": {
        "actual_winner": "SNP", "actual_majority": 4731,
        "actual_majority_pp": 15.51, "turnout_pct": 47.44,
        "actual_shares": {"SNP": 38.56, "Reform": 23.05, "Conservative": 14.74, "Labour": 12.81, "LibDem": 9.26},
    },
    "Aberdeenshire East": {
        "actual_winner": "SNP", "actual_majority": 943,
        "actual_majority_pp": 2.71, "turnout_pct": 55.63,
        "actual_shares": {"SNP": 33.40, "Conservative": 30.69, "Reform": 20.14, "LibDem": 11.49, "Labour": 4.27},
    },
    "Airdrie": {
        "actual_winner": "SNP", "actual_majority": 4510,
        "actual_majority_pp": 17.70, "turnout_pct": 46.76,
        "actual_shares": {"SNP": 42.03, "Labour": 24.33, "Reform": 22.84, "Conservative": 4.49, "LibDem": 2.59},
    },
    "Almond Valley": {
        "actual_winner": "SNP", "actual_majority": 8909,
        "actual_majority_pp": 24.33, "turnout_pct": 50.33,
        "actual_shares": {"SNP": 46.27, "Labour": 21.94, "Reform": 18.65, "Conservative": 6.83, "LibDem": 6.30},
    },
    "Angus North and Mearns": {
        "actual_winner": "SNP", "actual_majority": 3250,
        "actual_majority_pp": 11.21, "turnout_pct": 52.66,
        "actual_shares": {"SNP": 38.99, "Conservative": 27.79, "Reform": 16.70, "LibDem": 9.13, "Labour": 5.78},
    },
    "Angus South": {
        "actual_winner": "SNP", "actual_majority": 6608,
        "actual_majority_pp": 21.05, "turnout_pct": 53.22,
        "actual_shares": {"SNP": 42.34, "Conservative": 21.29, "Reform": 17.79, "Labour": 10.65, "LibDem": 7.93},
    },
    "Argyll and Bute": {
        "actual_winner": "SNP", "actual_majority": 2551,
        "actual_majority_pp": 9.26, "turnout_pct": 55.34,
        "actual_shares": {"SNP": 39.99, "LibDem": 30.73, "Reform": 13.35, "Labour": 6.31, "Conservative": 6.18},
    },
    "Ayr": {
        "actual_winner": "SNP", "actual_majority": 4400,
        "actual_majority_pp": 12.47, "turnout_pct": 56.02,
        "actual_shares": {"SNP": 36.40, "Conservative": 23.94, "Labour": 18.12, "Reform": 15.17, "LibDem": 4.04},
    },
    "Banffshire and Buchan Coast": {
        "actual_winner": "SNP", "actual_majority": 364,
        "actual_majority_pp": 1.23, "turnout_pct": 47.94,
        "actual_shares": {"SNP": 35.17, "Reform": 33.93, "Conservative": 21.52, "LibDem": 3.94, "Labour": 3.56},
    },
    "Bathgate": {
        "actual_winner": "SNP", "actual_majority": 5587,
        "actual_majority_pp": 16.74, "turnout_pct": 49.03,
        "actual_shares": {"SNP": 40.74, "Labour": 24.00, "Reform": 22.51, "Conservative": 6.27, "LibDem": 5.81},
    },
    "Caithness Sutherland and Ross": {
        "actual_winner": "LibDem", "actual_majority": 5092,
        "actual_majority_pp": 16.68, "turnout_pct": 53.99,
        "actual_shares": {"LibDem": 48.04, "SNP": 31.36, "Reform": 12.78, "Conservative": 3.66, "Labour": 2.93},
    },
    "Carrick Cumnock and Doon Valley": {
        "actual_winner": "SNP", "actual_majority": 2622,
        "actual_majority_pp": 9.05, "turnout_pct": 48.74,
        "actual_shares": {"SNP": 33.18, "Reform": 24.13, "Labour": 23.03, "Conservative": 12.71, "LibDem": 4.10},
    },
    "Clackmannanshire and Dunblane": {
        "actual_winner": "SNP", "actual_majority": 4264,
        "actual_majority_pp": 13.17, "turnout_pct": 56.67,
        "actual_shares": {"SNP": 37.76, "Labour": 24.59, "Reform": 16.01, "Conservative": 11.10, "LibDem": 5.69},
    },
    "Clydebank and Milngavie": {
        "actual_winner": "SNP", "actual_majority": 4197,
        "actual_majority_pp": 13.34, "turnout_pct": 57.73,
        "actual_shares": {"SNP": 38.56, "Labour": 25.21, "Reform": 14.34, "LibDem": 14.05, "Conservative": 5.79},
    },
    "Clydesdale": {
        "actual_winner": "SNP", "actual_majority": 4388,
        "actual_majority_pp": 12.25, "turnout_pct": 56.55,
        "actual_shares": {"SNP": 36.31, "Labour": 24.06, "Reform": 22.05, "Conservative": 12.13, "LibDem": 5.46},
    },
    "Coatbridge and Chryston": {
        "actual_winner": "SNP", "actual_majority": 6776,
        "actual_majority_pp": 23.01, "turnout_pct": 49.88,
        "actual_shares": {"SNP": 49.11, "Labour": 26.09, "Reform": 17.48, "Conservative": 3.77, "LibDem": 3.56},
    },
    "Cowdenbeath": {
        "actual_winner": "SNP", "actual_majority": 5687,
        "actual_majority_pp": 21.02, "turnout_pct": 47.29,
        "actual_shares": {"SNP": 44.32, "Labour": 23.31, "Reform": 17.40, "Conservative": 8.13, "LibDem": 5.72},
    },
    "Cumbernauld and Kilsyth": {
        "actual_winner": "SNP", "actual_majority": 7315,
        "actual_majority_pp": 26.95, "turnout_pct": 53.40,
        "actual_shares": {"SNP": 50.79, "Labour": 23.84, "Reform": 16.87, "Conservative": 4.16, "LibDem": 3.50},
    },
    "Cunninghame North": {
        "actual_winner": "SNP", "actual_majority": 5792,
        "actual_majority_pp": 19.21, "turnout_pct": 53.25,
        "actual_shares": {"SNP": 39.19, "Labour": 19.98, "Reform": 17.93, "Conservative": 16.27, "LibDem": 5.28},
    },
    "Cunninghame South": {
        "actual_winner": "SNP", "actual_majority": 4167,
        "actual_majority_pp": 14.22, "turnout_pct": 48.62,
        "actual_shares": {"SNP": 38.83, "Labour": 24.60, "Reform": 24.06, "Conservative": 7.58, "LibDem": 4.92},
    },
    "Dumfriesshire": {
        "actual_winner": "Conservative", "actual_majority": 1108,
        "actual_majority_pp": 3.39, "turnout_pct": 51.55,
        "actual_shares": {"Conservative": 34.84, "SNP": 31.44, "Reform": 17.72, "Labour": 10.31, "LibDem": 5.09},
    },
    "Dundee City East": {
        "actual_winner": "SNP", "actual_majority": 8177,
        "actual_majority_pp": 30.80, "turnout_pct": 46.50,
        "actual_shares": {"SNP": 48.85, "Labour": 18.05, "Reform": 15.57, "LibDem": 7.86, "Conservative": 7.53},
    },
    "Dundee City West": {
        "actual_winner": "SNP", "actual_majority": 6357,
        "actual_majority_pp": 24.53, "turnout_pct": 46.19,
        "actual_shares": {"SNP": 49.10, "Labour": 24.56, "Reform": 12.79, "LibDem": 7.64, "Conservative": 3.40},
    },
    "Dunfermline": {
        "actual_winner": "SNP", "actual_majority": 5437,
        "actual_majority_pp": 15.93, "turnout_pct": 53.12,
        "actual_shares": {"SNP": 41.63, "Labour": 25.70, "Reform": 14.92, "LibDem": 11.28, "Conservative": 6.47},
    },
    "East Kilbride": {
        "actual_winner": "SNP", "actual_majority": 4944,
        "actual_majority_pp": 14.62, "turnout_pct": 55.40,
        "actual_shares": {"SNP": 42.41, "Labour": 27.79, "Reform": 16.81, "Conservative": 6.66, "LibDem": 4.21},
    },
    "East Lothian Coast and Lammermuirs": {
        "actual_winner": "SNP", "actual_majority": 418,
        "actual_majority_pp": 1.17, "turnout_pct": 59.66,
        "actual_shares": {"SNP": 32.74, "Labour": 31.57, "Conservative": 13.23, "Reform": 12.93, "LibDem": 7.86},
    },
    "Eastwood": {
        "actual_winner": "SNP", "actual_majority": 732,
        "actual_majority_pp": 1.91, "turnout_pct": 65.59,
        "actual_shares": {"SNP": 33.23, "Conservative": 31.32, "Labour": 21.86, "Reform": 9.02, "LibDem": 4.57},
    },
    "Edinburgh Eastern Musselburgh and Tranent": {
        "actual_winner": "SNP", "actual_majority": 4986,
        "actual_majority_pp": 15.84, "turnout_pct": 51.86,
        "actual_shares": {"SNP": 44.73, "Labour": 28.90, "Reform": 13.09, "LibDem": 6.53, "Conservative": 5.78},
    },
    "Edinburgh North Eastern and Leith": {
        "actual_winner": "SNP", "actual_majority": 3071,
        "actual_majority_pp": 8.08, "turnout_pct": 56.97,
        "actual_shares": {"SNP": 35.85, "Green": 27.77, "Labour": 20.76, "Reform": 7.22, "LibDem": 4.98},
    },
    "Edinburgh North Western": {
        "actual_winner": "LibDem", "actual_majority": 13016,
        "actual_majority_pp": 32.43, "turnout_pct": 62.49,
        "actual_shares": {"LibDem": 57.20, "SNP": 24.77, "Reform": 8.33, "Labour": 4.68, "Conservative": 4.36},
    },
    "Edinburgh South Western": {
        "actual_winner": "SNP", "actual_majority": 3289,
        "actual_majority_pp": 10.15, "turnout_pct": 53.28,
        "actual_shares": {"SNP": 36.18, "Labour": 26.04, "Conservative": 14.30, "Reform": 12.14, "LibDem": 11.33},
    },
    "Ettrick Roxburgh and Berwickshire": {
        "actual_winner": "Conservative", "actual_majority": 5277,
        "actual_majority_pp": 17.53, "turnout_pct": 52.76,
        "actual_shares": {"Conservative": 44.80, "SNP": 27.26, "Reform": 11.86, "LibDem": 7.83, "Labour": 5.24},
    },
    "Falkirk East and Linlithgow": {
        "actual_winner": "SNP", "actual_majority": 5435,
        "actual_majority_pp": 14.48, "turnout_pct": 55.08,
        "actual_shares": {"SNP": 38.55, "Labour": 24.07, "Reform": 21.07, "Conservative": 7.62, "LibDem": 6.67},
    },
    "Falkirk West": {
        "actual_winner": "SNP", "actual_majority": 6736,
        "actual_majority_pp": 18.65, "turnout_pct": 52.07,
        "actual_shares": {"SNP": 41.25, "Reform": 22.60, "Labour": 21.76, "Conservative": 6.36, "LibDem": 5.64},
    },
    "Fife Mid and Glenrothes": {
        "actual_winner": "SNP", "actual_majority": 7634,
        "actual_majority_pp": 29.68, "turnout_pct": 46.41,
        "actual_shares": {"SNP": 48.45, "Reform": 18.77, "Labour": 15.44, "LibDem": 11.85, "Conservative": 5.50},
    },
    "Galloway and Dumfries West": {
        "actual_winner": "Conservative", "actual_majority": 1599,
        "actual_majority_pp": 5.33, "turnout_pct": 51.68,
        "actual_shares": {"Conservative": 38.34, "SNP": 33.01, "Reform": 15.58, "Labour": 8.48, "LibDem": 4.60},
    },
    "Glasgow Anniesland": {
        "actual_winner": "SNP", "actual_majority": 4659,
        "actual_majority_pp": 15.07, "turnout_pct": 51.95,
        "actual_shares": {"SNP": 44.71, "Labour": 29.64, "Reform": 15.65, "LibDem": 5.46, "Conservative": 4.54},
    },
    "Glasgow Baillieston and Shettleston": {
        "actual_winner": "SNP", "actual_majority": 5103,
        "actual_majority_pp": 18.89, "turnout_pct": 46.95,
        "actual_shares": {"SNP": 44.69, "Reform": 25.80, "Labour": 21.78, "LibDem": 4.00, "Conservative": 3.72},
    },
    "Glasgow Cathcart and Pollok": {
        "actual_winner": "SNP", "actual_majority": 5163,
        "actual_majority_pp": 16.05, "turnout_pct": 49.38,
        "actual_shares": {"SNP": 44.35, "Labour": 28.30, "Reform": 16.53, "LibDem": 4.37, "Conservative": 4.12},
    },
    "Glasgow Central": {
        "actual_winner": "SNP", "actual_majority": 9991,
        "actual_majority_pp": 38.04, "turnout_pct": 44.86,
        "actual_shares": {"SNP": 57.44, "Labour": 19.40, "Reform": 15.18, "LibDem": 4.81, "Conservative": 3.18},
    },
    "Glasgow Easterhouse and Springburn": {
        "actual_winner": "SNP", "actual_majority": 5154,
        "actual_majority_pp": 19.98, "turnout_pct": 43.33,
        "actual_shares": {"SNP": 46.22, "Labour": 26.25, "Reform": 20.58, "LibDem": 3.05, "Conservative": 2.74},
    },
    "Glasgow Kelvin and Maryhill": {
        "actual_winner": "SNP", "actual_majority": 2171,
        "actual_majority_pp": 6.37, "turnout_pct": 55.38,
        "actual_shares": {"SNP": 32.80, "Green": 26.43, "Labour": 22.11, "Reform": 10.55, "LibDem": 3.89},
    },
    "Hamilton Larkhall and Stonehouse": {
        "actual_winner": "SNP", "actual_majority": 2705,
        "actual_majority_pp": 8.65, "turnout_pct": 50.18,
        "actual_shares": {"SNP": 37.83, "Labour": 29.18, "Reform": 23.01, "Conservative": 5.17, "LibDem": 3.70},
    },
    "Kilmarnock and Irvine Valley": {
        "actual_winner": "SNP", "actual_majority": 4461,
        "actual_majority_pp": 15.29, "turnout_pct": 50.35,
        "actual_shares": {"SNP": 40.85, "Labour": 25.56, "Reform": 18.65, "Conservative": 9.53, "LibDem": 4.01},
    },
    "Kirkcaldy": {
        "actual_winner": "SNP", "actual_majority": 4747,
        "actual_majority_pp": 16.80, "turnout_pct": 45.27,
        "actual_shares": {"SNP": 43.29, "Labour": 26.49, "Reform": 19.41, "LibDem": 5.69, "Conservative": 5.12},
    },
    "Midlothian North": {
        "actual_winner": "SNP", "actual_majority": 2496,
        "actual_majority_pp": 8.63, "turnout_pct": 50.98,
        "actual_shares": {"SNP": 38.89, "Labour": 30.26, "Reform": 15.58, "LibDem": 8.25, "Conservative": 7.02},
    },
    "Midlothian South Tweeddale and Lauderdale": {
        "actual_winner": "SNP", "actual_majority": 7161,
        "actual_majority_pp": 20.77, "turnout_pct": 55.20,
        "actual_shares": {"SNP": 40.86, "Conservative": 20.10, "LibDem": 13.48, "Labour": 13.38, "Reform": 12.18},
    },
    "Moray": {
        "actual_winner": "SNP", "actual_majority": 2673,
        "actual_majority_pp": 8.15, "turnout_pct": 50.81,
        "actual_shares": {"SNP": 38.51, "Conservative": 30.36, "Reform": 16.58, "Labour": 7.30, "LibDem": 6.29},
    },
    "Motherwell and Wishaw": {
        "actual_winner": "SNP", "actual_majority": 5515,
        "actual_majority_pp": 20.17, "turnout_pct": 47.27,
        "actual_shares": {"SNP": 43.75, "Labour": 23.58, "Reform": 20.82, "Conservative": 5.99, "LibDem": 3.58},
    },
    "Paisley": {
        "actual_winner": "SNP", "actual_majority": 3028,
        "actual_majority_pp": 9.80, "turnout_pct": 50.33,
        "actual_shares": {"SNP": 42.60, "Labour": 32.80, "Reform": 14.95, "LibDem": 3.47, "Conservative": 3.04},
    },
    "Perthshire North": {
        "actual_winner": "SNP", "actual_majority": 6243,
        "actual_majority_pp": 17.25, "turnout_pct": 58.12,
        "actual_shares": {"SNP": 45.36, "Conservative": 28.11, "Reform": 12.77, "LibDem": 7.57, "Labour": 6.19},
    },
    "Perthshire South and Kinross-shire": {
        "actual_winner": "SNP", "actual_majority": 5061,
        "actual_majority_pp": 13.90, "turnout_pct": 58.91,
        "actual_shares": {"SNP": 40.39, "Conservative": 26.49, "Reform": 14.08, "LibDem": 11.89, "Labour": 7.14},
    },
    "Renfrewshire North and Cardonald": {
        "actual_winner": "SNP", "actual_majority": 4876,
        "actual_majority_pp": 13.94, "turnout_pct": 53.51,
        "actual_shares": {"SNP": 40.87, "Labour": 26.93, "Reform": 20.24, "Conservative": 5.51, "LibDem": 4.69},
    },
    "Renfrewshire West and Levern Valley": {
        "actual_winner": "SNP", "actual_majority": 3271,
        "actual_majority_pp": 9.56, "turnout_pct": 53.83,
        "actual_shares": {"SNP": 40.39, "Labour": 30.83, "Reform": 17.08, "Conservative": 5.89, "LibDem": 4.71},
    },
    "Rutherglen and Cambuslang": {
        "actual_winner": "SNP", "actual_majority": 5844,
        "actual_majority_pp": 17.25, "turnout_pct": 52.21,
        "actual_shares": {"SNP": 44.18, "Labour": 26.93, "Reform": 18.20, "LibDem": 5.41, "Conservative": 3.90},
    },
    "Shetland Islands": {
        "actual_winner": "SNP", "actual_majority": 1517,
        "actual_majority_pp": 13.21, "turnout_pct": 63.87,
        "actual_shares": {"SNP": 47.48, "LibDem": 34.27, "Green": 8.26, "Reform": 6.31, "Labour": 1.47},
    },
    "Skye Lochaber and Badenoch": {
        "actual_winner": "LibDem", "actual_majority": 952,
        "actual_majority_pp": 2.43, "turnout_pct": 57.47,
        "actual_shares": {"LibDem": 38.88, "SNP": 36.45, "Reform": 11.93, "Conservative": 6.98, "Labour": 4.47},
    },
    "Stirling": {
        "actual_winner": "SNP", "actual_majority": 7442,
        "actual_majority_pp": 23.16, "turnout_pct": 58.06,
        "actual_shares": {"SNP": 42.35, "Conservative": 19.19, "Labour": 15.95, "Reform": 15.28, "LibDem": 7.23},
    },
    "Uddingston and Bellshill": {
        "actual_winner": "SNP", "actual_majority": 3584,
        "actual_majority_pp": 12.43, "turnout_pct": 48.67,
        "actual_shares": {"SNP": 41.49, "Labour": 29.07, "Reform": 19.58, "Conservative": 6.07, "LibDem": 3.79},
    },
}

st.set_page_config(page_title="Constituency Explorer", page_icon="🗺️", layout="wide")
st.title("🗺️ Constituency Explorer")
st.markdown(
    "Search any of the 73 Scottish Parliament constituency seats. "
    "Results update as you type."
)

# ── Search input ──────────────────────────────────────────────────────────────
search_term = st.text_input("Search constituency name...", placeholder="e.g. Glasgow, Orkney, Edinburgh")

# ── Fetch from API ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner="Fetching constituency data…")
def fetch_constituencies(search: str) -> dict | None:
    params = {}
    if search.strip():
        params["search"] = search.strip()
    try:
        resp = requests.get(
            f"{API_BASE}/seats/constituencies",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Could not connect to API at {API_BASE}. Is the API container running?"}
    except requests.exceptions.Timeout:
        return {"error": "API request timed out."}
    except requests.exceptions.HTTPError as exc:
        return {"error": f"API returned {exc.response.status_code}: {exc.response.text}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


data = fetch_constituencies(search_term)

if data is None or "error" in data:
    st.error(data["error"] if data else "Unknown error fetching data.")
    st.stop()

constituencies = data.get("constituencies", [])
n_results = data.get("n_results", len(constituencies))

# ── Summary metrics ───────────────────────────────────────────────────────────
n_marginal = sum(1 for c in constituencies if c["is_marginal"])

col1, col2 = st.columns(2)
col1.metric("Results", n_results)
col2.metric("Marginal seats in results", n_marginal)

st.divider()

if not constituencies:
    st.info("No constituencies match your search.")
    st.stop()

# ── Build display dataframe ───────────────────────────────────────────────────
def _top_dev_str(api_row: dict, actual: dict | None) -> str:
    if not actual:
        return "—"
    pred_pct = {p: v * 100 for p, v in api_row["vote_shares"].items()}
    all_parties = set(pred_pct) | set(actual["actual_shares"])
    max_dev = max(
        (pred_pct.get(p, 0.0) - actual["actual_shares"].get(p, 0.0) for p in all_parties),
        key=abs,
    )
    return f"{max_dev:+.1f}"


rows = []
for c in constituencies:
    actual = ACTUAL_RESULTS.get(c["constituency"])
    rows.append({
        "Constituency": c["constituency"],
        "Region": c["region"],
        "Predicted Winner": c["predicted_winner"],
        "Majority Margin (pp)": c["majority_margin_pp"],
        "Marginal?": "Yes" if c["is_marginal"] else "No",
        "Tactical Rec.": c["tactical_vote_recommendation"] or "—",
        "Actual Winner": actual["actual_winner"] if actual else "—",
        "Actual Majority (pp)": f"{actual['actual_majority_pp']:.2f}" if actual else "—",
        "Turnout %": f"{actual['turnout_pct']:.1f}" if actual else "—",
        "Model Correct?": (
            "✅" if actual and actual["actual_winner"] == c["predicted_winner"]
            else ("❌" if actual else "—")
        ),
        "Top Deviation (pp)": _top_dev_str(c, actual),
    })
df = pd.DataFrame(rows)


def _style_row(row: pd.Series) -> list[str]:
    if row["Marginal?"] == "Yes":
        return ["background-color: #FEFBD0"] * len(row)
    return [""] * len(row)


def _colour_winner(val: str) -> str:
    colour = PARTY_COLOURS.get(val, "#888888")
    return f"background-color: {colour}33; color: {colour}; font-weight: bold"


def _colour_correct(val: str) -> str:
    if val == "✅":
        return "background-color:#D5F5E3; color:#1E8449; font-weight:bold"
    if val == "❌":
        return "background-color:#FADBD8; color:#C0392B; font-weight:bold"
    return ""


def _colour_top_dev(val: str) -> str:
    try:
        if abs(float(val)) > 10:
            return "background-color:#FADBD8; color:#C0392B; font-weight:bold"
    except (ValueError, TypeError):
        pass
    return ""


def _render_detail_card(c: dict) -> None:
    """Render the full detail card for a single constituency API result dict."""
    st.divider()
    st.markdown(f"### 📌 {c['constituency']}")

    info_col, chart_col = st.columns([1, 2])
    with info_col:
        st.markdown(f"**Region:** {c['region']}")
        _pw = c["predicted_winner"]
        _pw_colour = PARTY_COLOURS.get(_pw, "#888")
        st.markdown(
            f"**Predicted winner:** "
            f"<span style='color:{_pw_colour}; font-weight:bold'>{_pw}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Majority margin:** {c['majority_margin_pp']:.2f} pp")
        st.markdown(f"**Marginal seat:** {'Yes ⚠️' if c['is_marginal'] else 'No'}")
        if c["tactical_vote_recommendation"]:
            st.markdown(f"**Tactical rec.:** {c['tactical_vote_recommendation']}")

    with chart_col:
        vote_shares: dict[str, float] = c["vote_shares"]
        sorted_items = sorted(vote_shares.items(), key=lambda x: x[1], reverse=True)
        parties = [p for p, _ in sorted_items]
        shares = [v * 100 for _, v in sorted_items]
        colours = [PARTY_COLOURS.get(p, "#888888") for p in parties]

        fig = go.Figure(
            go.Bar(
                x=parties,
                y=shares,
                marker_color=colours,
                text=[f"{s:.1f}%" for s in shares],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Vote share: %{y:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title="Predicted vote shares",
            yaxis_title="Vote share (%)",
            yaxis_range=[0, max(shares) * 1.2],
            height=320,
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Actual 2026 results ───────────────────────────────────────────────────
    actual = ACTUAL_RESULTS.get(c["constituency"])
    if actual:
        st.divider()
        predicted_winner = c["predicted_winner"]
        actual_winner = actual["actual_winner"]

        if predicted_winner != actual_winner:
            st.warning(
                f"⚠️ Model predicted **{predicted_winner}** — "
                f"actual winner was **{actual_winner}**"
            )

        st.subheader("🗳️ Actual Result (8 May 2026)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Actual Winner", actual_winner)
        m2.metric("Actual Majority", f"{actual['actual_majority_pp']:.2f} pp")
        m3.metric("Turnout", f"{actual['turnout_pct']:.1f}%")

        act_items = sorted(actual["actual_shares"].items(), key=lambda x: x[1], reverse=True)
        act_parties = [p for p, _ in act_items]
        act_shares_vals = [v for _, v in act_items]
        act_colours = [PARTY_COLOURS.get(p, "#888888") for p in act_parties]

        fig_act = go.Figure(
            go.Bar(
                x=act_parties,
                y=act_shares_vals,
                marker_color=act_colours,
                text=[f"{v:.1f}%" for v in act_shares_vals],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Vote share: %{y:.1f}%<extra></extra>",
            )
        )
        fig_act.update_layout(
            title="Actual vote shares (%)",
            yaxis_title="Vote share (%)",
            yaxis_range=[0, max(act_shares_vals) * 1.2],
            height=320,
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_act, use_container_width=True)

        st.subheader("📊 Model vs Actual Deviation")
        pred_shares_pct: dict[str, float] = {p: v * 100 for p, v in c["vote_shares"].items()}
        all_parties_dev = sorted(
            set(pred_shares_pct) | set(actual["actual_shares"]),
            key=lambda p: abs(
                pred_shares_pct.get(p, 0.0) - actual["actual_shares"].get(p, 0.0)
            ),
            reverse=True,
        )
        dev_rows = [
            {
                "Party": p,
                "Predicted %": pred_shares_pct.get(p, 0.0),
                "Actual %": actual["actual_shares"].get(p, 0.0),
                "Deviation (pp)": (
                    pred_shares_pct.get(p, 0.0) - actual["actual_shares"].get(p, 0.0)
                ),
            }
            for p in all_parties_dev
        ]
        dev_df = pd.DataFrame(dev_rows)

        def _colour_deviation(val: float) -> str:
            if abs(val) > 5:
                return "background-color:#FADBD8; color:#C0392B"
            if abs(val) >= 2:
                return "background-color:#FDEBD0; color:#E67E22"
            return "background-color:#D5F5E3; color:#1E8449"

        dev_styled = (
            dev_df.style
            .map(_colour_deviation, subset=["Deviation (pp)"])
            .format({"Predicted %": "{:.1f}", "Actual %": "{:.1f}", "Deviation (pp)": "{:+.2f}"})
            .set_properties(**{"font-size": "13px"})
        )
        st.dataframe(dev_styled, use_container_width=True, hide_index=True)
        st.caption("Positive = model overestimated; Negative = model underestimated")
    else:
        st.divider()
        st.info(
            "Actual 2026 results not yet loaded for this constituency. "
            "Full results: [ELECTION_RESULTS.md](ELECTION_RESULTS.md)"
        )


styled = (
    df.style.apply(_style_row, axis=1)
    .map(_colour_winner, subset=["Predicted Winner"])
    .map(_colour_winner, subset=["Actual Winner"])
    .map(_colour_correct, subset=["Model Correct?"])
    .map(_colour_top_dev, subset=["Top Deviation (pp)"])
    .format({"Majority Margin (pp)": "{:.2f}"})
    .set_properties(**{"font-size": "13px"})
)

st.markdown("### Results")
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Detail card (single result only) ─────────────────────────────────────────
if n_results == 1:
    _render_detail_card(constituencies[0])

st.divider()
st.caption("Built by **Debabrata Mishra** · Data Scientist / ML Engineer · "
           "[GitHub](https://github.com/dmishra27) · "
           "[LinkedIn](https://linkedin.com/in/debabrata-mishra) · "
           "[Portfolio](https://bit.ly/m/Debabrata_Mishra_Data_Science_Bio)")
