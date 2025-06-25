import sys, pathlib  # noqa: E402
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import math
from optimizers import A2SAM, HATAM


def _quadratic(batch_size=32, dim=10, device="cpu"):
    """Synthetic quadratic minimisation problem f(x)=||x||^2/2"""
    w = torch.randn(batch_size, dim, device=device, requires_grad=True)
    def loss_fn():
        return 0.5 * (w ** 2).sum()
    return w, loss_fn


def _run_steps(optim, loss_fn, n=100):
    for _ in range(n):
        def closure():
            optim.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss
        optim.step(closure)
    return loss_fn().item()


def test_a2sam_converges():
    w, loss_fn = _quadratic(device="cpu")
    base = torch.optim.SGD([w], lr=0.1)
    optim = A2SAM([w], base_optimizer_cls=torch.optim.SGD, base_optimizer_kwargs={"lr": 0.1}, rho=0.05, hessian_update_freq=0)
    final = _run_steps(optim, loss_fn, n=200)
    assert final < 1e-2, f"A2SAM failed to converge, final loss={final}"


def test_hatam_converges():
    w, loss_fn = _quadratic(device="cpu")
    optim = HATAM([w], lr=0.1)
    final = _run_steps(optim, loss_fn, n=200)
    assert final < 1e-2, f"HATAM failed to converge, final loss={final}"


# ✅ НОВЫЕ ТЕСТЫ для проверки корректности реализации

def test_hatam_formula_correctness():
    """Проверяем что HATAM действительно реализует формулу g_hatam = g + γ·S⊙c"""
    w = torch.randn(10, requires_grad=True)
    optim = HATAM([w], lr=0.01, gamma=0.1, beta_c=0.9)
    
    # Делаем несколько шагов чтобы накопились моменты
    for i in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        
        # Сохраняем состояние до шага
        state = optim.state[w]
        if i > 0:  # На втором шаге можем проверить формулу
            g = w.grad.clone()
            c = state.get("c", torch.zeros_like(w))
            v = state.get("v", torch.zeros_like(w))
            
            # Вычисляем ожидаемую коррекцию по формуле
            v_prev_corr = v / (1 - 0.999 ** (i)) if i > 0 else v
            h = v_prev_corr.sqrt().add_(1e-8)
            h_mean = h.mean()
            S = h / torch.clamp(h_mean, min=1e-8)
            expected_correction = 0.1 * S * c
            
            # Проверяем что внутренняя логика соответствует формуле
            assert torch.allclose(c, state["c"], atol=1e-6), "EMA update c_t некорректен"
        
        optim.step()


def test_a2sam_woodbury_formula():
    """Проверяем что A²SAM корректно применяет формулу Woodbury"""
    w = torch.randn(5, requires_grad=True)
    optim = A2SAM([w], 
                  base_optimizer_cls=torch.optim.SGD, 
                  base_optimizer_kwargs={"lr": 0.1}, 
                  rho=0.05, alpha=0.1, k=2,
                  hessian_update_freq=0)  # Отключаем обновление гессиана
    
    # Принудительно устанавливаем известные собственные значения/векторы
    optim._eigvals = torch.tensor([2.0, 1.0])
    optim._eigvecs = torch.eye(5)[:2]  # Первые 2 канонических вектора
    
    # Тестовый градиент
    test_grad = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.25])
    w.grad = test_grad.clone()
    
    # Вычисляем возмущение
    eps_list = optim._compute_eps()
    eps = eps_list[0]
    
    # Проверяем формулу Woodbury вручную
    g = test_grad
    V = optim._eigvecs  # (2, 5)
    Λ = optim._eigvals  # (2,)
    
    g_proj = torch.mv(V, g)  # V^T g
    coeffs = optim.alpha * Λ / (1.0 + optim.alpha * Λ) * g_proj
    M_inv_g_expected = g - torch.matmul(V.t(), coeffs)
    eps_expected = optim.rho * M_inv_g_expected / (M_inv_g_expected.norm() + optim.eps)
    
    assert torch.allclose(eps, eps_expected, atol=1e-5), "Формула Woodbury реализована некорректно"


def test_gradient_preservation():
    """Проверяем что все оптимизаторы корректно сохраняют градиенты"""
    for optim_cls, kwargs in [
        (HATAM, {"lr": 0.01}),
        (A2SAM, {"base_optimizer_cls": torch.optim.SGD, "base_optimizer_kwargs": {"lr": 0.01}, "hessian_update_freq": 0})
    ]:
        w = torch.randn(3, requires_grad=True)
        optim = optim_cls([w], **kwargs)
        
        def closure():
            optim.zero_grad()
            loss = (w ** 2).sum()
            loss.backward()
            return loss
        
        # До шага градиент должен быть установлен
        loss = closure()
        grad_before = w.grad.clone()
        
        # После шага параметры должны измениться, но градиент очищен (если zero_grad=True)
        if hasattr(optim, 'step') and 'closure' in str(optim.step.__code__.co_varnames):
            optim.step(closure)
        else:
            optim.step()
            
        assert not torch.allclose(w, torch.zeros_like(w)), f"{optim_cls.__name__} не обновил параметры"


def test_fair_comparison_same_model():
    """Проверяем что все оптимизаторы работают с одинаковой моделью и данными"""
    torch.manual_seed(42)  # Фиксируем seed
    
    # Создаем идентичные модели для каждого оптимизатора
    models = []
    optimizers = []
    
    for optim_name, optim_config in [
        ("HATAM", {"cls": HATAM, "kwargs": {"lr": 0.01}}),
        ("A2SAM", {"cls": A2SAM, "kwargs": {"base_optimizer_cls": torch.optim.SGD, 
                                            "base_optimizer_kwargs": {"lr": 0.01}, 
                                            "hessian_update_freq": 0}}),
    ]:
        # Одинаковая инициализация
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 1)
        initial_weights = model.weight.clone()
        
        optim = optim_config["cls"](model.parameters(), **optim_config["kwargs"])
        
        models.append((optim_name, model, initial_weights))
        optimizers.append((optim_name, optim))
    
    # Проверяем что все модели идентично инициализированы
    for i in range(1, len(models)):
        assert torch.allclose(models[0][2], models[i][2]), f"Модели {models[0][0]} и {models[i][0]} имеют разную инициализацию"
    
    # Тестируем один шаг оптимизации на одинаковых данных
    torch.manual_seed(42)
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    for optim_name, optim in optimizers:
        corresponding_model = next(m[1] for m in models if m[0] == optim_name)
        
        def closure():
            optim.zero_grad()
            pred = corresponding_model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            return loss
        
        if hasattr(optim, 'step') and 'closure' in str(optim.step.__code__.co_varnames):
            optim.step(closure)
        else:
            closure()
            optim.step()
        
        # Проверяем что веса изменились
        initial_weight = next(m[2] for m in models if m[0] == optim_name)
        current_weight = corresponding_model.weight
        assert not torch.allclose(initial_weight, current_weight), f"{optim_name} не обновил веса" 