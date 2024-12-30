import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def inverse_y(y, y_scaler=None):
    if y_scaler is None:
        y_scaler: StandardScaler = getattr(sys.modules['__main__'], 'y_scaler')
    if isinstance(y, pd.Series):
        return np.expm1(y_scaler.inverse_transform(y.values.reshape(-1,1)).reshape(-1))
    else:
        return np.expm1(y_scaler.inverse_transform(y.reshape(-1,1)).reshape(-1))
    

# for lgboost
from typing import Tuple

def gradient(
    preds: np.ndarray,
    labels: np.ndarray, 
) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    return (np.log1p(preds) - np.log1p(labels)) / (preds + 1)

def hessian(
    preds: np.ndarray,
    labels: np.ndarray, 
) -> np.ndarray:
    '''Compute the hessian for squared log error.'''

    return ((-np.log1p(preds) + np.log1p(labels) + 1) /
            np.square(preds + 1))

def sle_grad_hess(
    preds: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    
    labels[labels < -1] = -1 + 1e-6
    grad = gradient(preds, labels)
    hess = hessian(preds, labels)
    return grad, hess

# def sle_grad_hess(yhat: np.ndarray, y: np.ndarray, weights: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
#     # when weights is None, it means that all weights are 1
#     if weights is None:
#         weights = np.ones(len(y))
        
#     eps = 1e-38 - 1
#     y = np.maximum(y, eps)  # Ensure (preds+1) > 0
#     yhat = np.maximum(yhat, eps)  # Ensure (labels+1) > 0
#     log1p_y = np.log1p(y)
#     log1p_yhat = np.log1p(yhat)

#     diff = (log1p_yhat - log1p_y)

#     grad = diff / (yhat + 1.0) * weights
#     hess = (1 - diff) / np.square(yhat + 1.0) * weights

#     return grad, hess


# for lgboost
def lgb_rmsle(
    preds: np.ndarray, 
    dtrain: np.ndarray,
):
    dtrain = np.log1p(dtrain)
    preds = np.log1p(preds)

    rmsle = np.sqrt(mean_squared_error(dtrain, preds))

    # print(rmsle)
    return 'RMSLE', rmsle, False

def lgb_lse_objective(
    dtrain: np.ndarray,
    preds: np.ndarray, 
):
    
    grad, hess = sle_grad_hess(preds, dtrain)
    return grad, hess

def lgb_rmsle_with_inverse(preds, dtrain: np.ndarray):
    preds = inverse_y(preds)
    dtrain = inverse_y(dtrain)
    return lgb_rmsle(preds, dtrain)

def lgb_lse_objective_with_inverse(preds, dtrain: np.ndarray):
    preds = inverse_y(preds)
    dtrain = inverse_y(dtrain)
    # g(y) = inverse_y(y) = np.expm1((y) * y_scaler.scale_+y_scaler.mean_)
    # loss(g(y))'' = loss''(g(y)) * g'(y)^2 + loss'(g(y)) * g''(y) 
    y_scaler = getattr(sys.modules['__main__'], 'y_scaler')
    g_prime = preds * y_scaler.scale_
    g_double_prime = g_prime * y_scaler.scale_
    grad, hess = lgb_lse_objective(preds, dtrain)
    return grad * g_prime, hess * np.square(g_prime) + grad * g_double_prime


# for catboost
class RmsleMetric(object):
    def __init__(self, use_inverse=True):
        self.inverse = use_inverse
    @staticmethod
    def is_max_optimal():
        # RMLSE는 값이 작을수록 좋으므로 False 반환
        return False

    def evaluate(self, approxes, target, weight=None):
        """
        RMLSE 계산 함수.
        
        Args:
            approxes: 예측값의 리스트 (CatBoost에서 제공)
            target: 실제 타겟값
            weight: 가중치 (없을 수 있음, None 처리 필요)
        
        Returns:
            (error, weights sum)
        """
        # 예측값 가져오기
        preds = np.array(approxes[0])  # approxes는 리스트로 제공됨
        target = np.array(target)
        
        # invesre transform
        if self.inverse:
            preds = inverse_y(preds)
            target = inverse_y(target)

        # 음수값 방지
        eps = 1e-38
        preds = np.maximum(preds, eps)  # Ensure (labels+1) > 0
        target = np.maximum(target, eps)  # Ensure (preds+1) > 0

        if np.any(target <= 0) or np.any(preds < 0):
            raise ValueError("RMLSE cannot be calculated for negative values.")

        # 로그 변환
        preds = np.log1p(preds)
        target = np.log1p(target)

        # 가중치 처리
        if weight is None:
            weight = np.ones(target.shape[0])
        
        error_sum = np.sum((weight * np.square(preds - target)))
        weight_sum = np.sum(weight)

        return error_sum, weight_sum

    @staticmethod
    def get_final_error(error_sum, weight_sum):
        """
        최종 에러 값 반환.
        
        Args:
            error_sum: RMLSE 값
            weight_sum: 가중치 합계
        
        Returns:
            최종 RMLSE 값
        """
        return np.sqrt(error_sum / (weight_sum + 1e-38))  # RMLSE 자체를 최종 결과로 반환
    


class SleObjective(object):
    def __init__(self, use_inverse):
        self.inverse = use_inverse
    def calc_ders_range(self, approxes: np.ndarray, targets: np.ndarray, weights=None):
        """
        RMSLE 손실 함수에 대한 Gradient 및 Hessian을 반환
        Args:
            approxes (list): 모델의 예측값 리스트
            targets (list): 실제 타겟값 리스트
            weights (list, optional): 가중치 리스트
        Returns:
            list of tuples: 각 데이터 포인트에 대해 (Gradient, Hessian) 반환
        """
        assert len(approxes) == len(targets), "Predictions and targets must have the same length"

        if self.inverse:
            approxes = inverse_y(approxes)
            targets = inverse_y(targets)
        
        grad, hess = sle_grad_hess(approxes, targets)
        if self.inverse:
            y_scaler = getattr(sys.modules['__main__'], 'y_scaler')
            g_prime = approxes * y_scaler.scale_
            g_double_prime = g_prime * y_scaler.scale_
            grad = grad * approxes * g_prime
            hess = hess * np.square(g_prime) + grad * g_double_prime
        return list(zip(grad, hess))
    
# for xgboost
def xgb_sle_objective(labels: np.ndarray, preds: np.ndarray):
    grad, hess = sle_grad_hess(preds, labels)
    return grad, hess

def xgb_sle_objective_with_inverse(labels, preds):
    y = inverse_y(labels)
    yhat = inverse_y(preds)
    grad, hess = xgb_sle_objective(y, yhat)
    y_scaler = getattr(sys.modules['__main__'], 'y_scaler')
    g_prime = yhat * y_scaler.scale_
    g_double_prime = g_prime * y_scaler.scale_
    return grad * g_prime, hess * np.square(g_prime) + grad * g_double_prime

def xgb_rmsle_evaluation_with_inverse(labels, preds):
    y = inverse_y(labels)
    yhat = inverse_y(preds)
    return xgb_rmsle_evaluation(y, yhat)


def xgb_rmsle_evaluation(labels: np.ndarray, preds: np.ndarray):
    y = labels
    yhat = preds
    n = y.shape[0]
    eps = 1e-6 - 1
    y = np.maximum(labels, eps)  # Ensure (preds+1) > 0
    yhat = np.maximum(preds, eps)  # Ensure (labels+1) > 0
    log1p_y = np.log1p(y)
    log1p_yhat = np.log1p(yhat)
    rmlse = np.sqrt(np.sum(np.square(log1p_yhat - log1p_y)) / n)
    return rmlse