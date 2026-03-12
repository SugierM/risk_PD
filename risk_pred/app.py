import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
from src.model_grade import FlexibleScoringNet
import torch
import pandas as pd
from src.utils import MAPPED_BINS, MAPPED_FILLS, prepare_for_pd, provide_inter
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

GRADE_MODEL_DIR = Path.cwd() / "models" / "grade_models"
PD_MODEL_DIR = Path.cwd() / "models" / "pd"

DEFAULT_GRADE = 0.454

PD_FIELDS = [
    {"name": "annual_inc", "label": "Dochód roczny", "type": "number", "required": True},
    {"name": "loan_amnt", "label": "Kwota pożyczki", "type": "number", "required": True},
    {"name": "dti", "label": "Zadłużenie do dochodu (%)", "type": "number", "step": "0.1", "required": True}, # wierd to ask it here, but stays for now
    {"name": "inq_last_6mths", "label": "Zapytania (w ciągu ostatnich 6 miesięcy)", "type": "number", "required": True},
    {"name": "total_rev_hi_lim", "label": "Limit na rachunkach odnawialnych", "type": "number", "required": True},
    {"name": "revol_util", "label": "Wykorzystanie limitów na rachunkach odnawialnych (%)", "type": "number", "required": True},
    {"name": "tot_cur_bal", "label": "Całkowite zadłużenie", "type": "number", "required": True},
    {"name": "bc_util", "label": "Wykorzystany limit na kartach kredytowych (%)", "type": "number", "required": True},
    {"name": "num_tl_op_past_12m", "label": "Liczba otwartych lini kredytowych w ostatnich 12m", "type": "number", "required": True},
    {"name": "mo_sin_old_rev_tl_op", "label": "Liczba miesięcy, od otwarcia najstarszego rachunku odnawialnego", "type": "number", "required": True},

    {
        "name": "home_ownership", 
        "label": "Typ własności", 
        "options": ["RENT", "MORTGAGE", "OWN_OTHER"]
    },
]

GRADE_FIELDS = [
    {"name": "term", "label": "Okres (msc)", "type": "number", "default": None},
    {"name": "purpose", "label": "Cel pożyczki", "type": "text", "default": None},
]

def load_all_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pd_dict = joblib.load(
        PD_MODEL_DIR / "logistic_regression_woe_v1.joblib"
    )
    model_grade_dict = joblib.load(GRADE_MODEL_DIR / "metadata.joblib")

    nn_scaler = model_grade_dict["scaler"]

    pd_order = model_pd_dict["feature_order"]
    grade_order = model_grade_dict["feature_order"]

    if not pd_order or not grade_order:
        raise ValueError(
            "Models need correct column order to perform how intended."
        )

    grade_input_dim = len(grade_order)
    best_arch_layers = (32,)
    best_model_path = (
        GRADE_MODEL_DIR / "grade_model_32_final.pth"
    )

    model_grade = FlexibleScoringNet(
        grade_input_dim,
        best_arch_layers,
    ).to(device)

    model_grade.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )
    model_grade.eval()

    return {
        "model_pd": model_pd_dict["model"],
        "mapped_bins": model_pd_dict["mapped_bins"],
        "woe_maps": model_pd_dict["woe_maps"],
        "mapped_fills": model_pd_dict["mapped_fills"],
        "pd_order": list(pd_order),
        "model_grade": model_grade,
        "grade_order": list(grade_order),
        "coefficients": model_pd_dict["coefficients"]
    }



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.assets = load_all_assets()
    yield
    app.state.assets.clear()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "pd_fields": PD_FIELDS,
        "grade_fields": GRADE_FIELDS
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()
    input_dict = dict(form_data)
    message = ""

    df_all = pd.DataFrame([input_dict])
    df_all = df_all.apply(pd.to_numeric, errors='ignore')
    assets = request.app.state.assets

    grade = DEFAULT_GRADE
    if all(input_dict.values()):
        df_grade = df_all[assets["grade_order"]]
        grade = DEFAULT_GRADE # FOR NOW

    df_all["probability"] = grade

    df_pd = prepare_for_pd(df_all, assets)
    verd = assets["model_pd"].predict_proba(df_pd)[:, 1][0]
    
    if verd > 0.5:
        interpret = provide_inter(df_pd, assets, input_dict, 3)
        message = "Najważniejsze powody udrzucenia:<br>"
        for p in interpret:
            message += f"<b>{p["feature"]}</b> - {p["impact"]}<br>"
            message += f"<b>Kategoria bazowa</b>: {p["base_cat"]}<br>"
            message += f"<b>Wartośc podana przez klienta</b>: {p["client_cat"]}<br>"
        

    result = {
        "decision": "Odmowa" if verd > 0.5 else "Zgoda",
        "probability": verd,
        "inter": message
    }

    return templates.TemplateResponse("form.html", {
        "request": request, 
        "pd_fields": PD_FIELDS,
        "grade_fields": GRADE_FIELDS,
        "result": result
    })


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)