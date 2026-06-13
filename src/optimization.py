from time import perf_counter

import numpy as np


def _init_history(oracle, x, start_time, trace):
    if not trace:
        return None
    return {
        "epoch": [0.0],
        "time": [0.0],
        "func": [oracle.func(x)],
        "grad_norm": [np.linalg.norm(oracle.grad(x))],
    }


def _record(history, oracle, x, epochs, start_time):
    if history is None:
        return
    history["epoch"].append(float(epochs))
    history["time"].append(float(perf_counter() - start_time))
    history["func"].append(oracle.func(x))
    history["grad_norm"].append(float(np.linalg.norm(oracle.grad(x))))


def _step_size(schedule, iteration, epoch, params):
    alpha0 = params.get("alpha_0", params.get("step_size", 1e-2))
    if schedule == "constant":
        return alpha0
    if schedule == "inverse_sqrt":
        return alpha0 / np.sqrt(iteration + 1.0)
    if schedule == "step_decay":
        gamma = params.get("gamma", 0.5)
        drop_freq = params.get("drop_freq", 10)
        return alpha0 * gamma ** np.floor(epoch / drop_freq)
    raise ValueError("Неизвестное расписание шага.")


def _iter_batches(m, batch_size, rng):
    perm = rng.permutation(m)
    for start in range(0, m, batch_size):
        yield perm[start : start + batch_size]


def gradient_descent(oracle, x_0, step_size=1e-2, max_epoch=100, trace=False, display=False):
    x = np.asarray(x_0, dtype=float).copy()
    start_time = perf_counter()
    history = _init_history(oracle, x, start_time, trace)

    for epoch in range(1, max_epoch + 1):
        x -= step_size * oracle.grad(x)
        _record(history, oracle, x, epoch, start_time)
        if display and epoch % 10 == 0:
            print(f"GD epoch={epoch}, F={history['func'][-1]:.6e}")

    return x, "success", history


def sgd(
    oracle,
    x_0,
    batch_size=32,
    max_epoch=100,
    lr_schedule="constant",
    lr_params=None,
    trace=False,
    display=False,
    random_state=42,
):
    """Mini-batch SGD с выбором батчей без возвращения внутри эпохи."""
    if lr_params is None:
        lr_params = {"alpha_0": 1e-2}

    rng = np.random.default_rng(random_state)
    x = np.asarray(x_0, dtype=float).copy()
    start_time = perf_counter()
    history = _init_history(oracle, x, start_time, trace)
    iteration = 0
    processed = 0
    next_log_epoch = 1

    while processed < max_epoch * oracle.m:
        for batch_idx in _iter_batches(oracle.m, batch_size, rng):
            epoch_float = processed / oracle.m
            alpha = _step_size(lr_schedule, iteration, epoch_float, lr_params)
            x -= alpha * oracle.grad(x, batch_idx)
            iteration += 1
            processed += batch_idx.size

            if processed / oracle.m >= next_log_epoch:
                _record(history, oracle, x, next_log_epoch, start_time)
                if display:
                    print(f"SGD epoch={next_log_epoch}, F={history['func'][-1]:.6e}, alpha={alpha:.3e}")
                next_log_epoch += 1
            if processed >= max_epoch * oracle.m:
                break

    return x, "success", history


def svrg(
    oracle,
    x_0,
    step_size=1e-2,
    batch_size=32,
    max_epoch=100,
    inner_epochs=1,
    trace=False,
    display=False,
    random_state=42,
):
    """SVRG с корректной обработкой L2-регуляризации."""
    rng = np.random.default_rng(random_state)
    x = np.asarray(x_0, dtype=float).copy()
    start_time = perf_counter()
    history = _init_history(oracle, x, start_time, trace)
    epochs_spent = 0.0
    next_log_epoch = 1

    while epochs_spent < max_epoch:
        x_tilde = x.copy()
        full_loss_grad_tilde = oracle.loss_grad(x_tilde)
        epochs_spent += 1.0
        if epochs_spent >= next_log_epoch:
            _record(history, oracle, x, next_log_epoch, start_time)
            next_log_epoch += 1

        inner_processed = 0
        target_inner = inner_epochs * oracle.m
        while inner_processed < target_inner and epochs_spent < max_epoch:
            batch_idx = rng.choice(oracle.m, size=min(batch_size, oracle.m), replace=False)
            grad_x = oracle.loss_grad(x, batch_idx)
            grad_tilde = oracle.loss_grad(x_tilde, batch_idx)
            grad_est = grad_x - grad_tilde + full_loss_grad_tilde + oracle.regcoef * x
            x -= step_size * grad_est

            cost = batch_idx.size / oracle.m
            inner_processed += batch_idx.size
            epochs_spent += cost
            if epochs_spent >= next_log_epoch:
                _record(history, oracle, x, next_log_epoch, start_time)
                if display:
                    print(f"SVRG epoch={next_log_epoch}, F={history['func'][-1]:.6e}")
                next_log_epoch += 1

    return x, "success", history


def adam(
    oracle,
    x_0,
    step_size=1e-3,
    batch_size=32,
    max_epoch=100,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    trace=False,
    display=False,
    random_state=42,
):
    rng = np.random.default_rng(random_state)
    x = np.asarray(x_0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    processed = 0
    next_log_epoch = 1
    start_time = perf_counter()
    history = _init_history(oracle, x, start_time, trace)

    while processed < max_epoch * oracle.m:
        for batch_idx in _iter_batches(oracle.m, batch_size, rng):
            g = oracle.grad(x, batch_idx)
            t += 1
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)
            x -= step_size * m_hat / (np.sqrt(v_hat) + eps)

            processed += batch_idx.size
            if processed / oracle.m >= next_log_epoch:
                _record(history, oracle, x, next_log_epoch, start_time)
                if display:
                    print(f"Adam epoch={next_log_epoch}, F={history['func'][-1]:.6e}")
                next_log_epoch += 1
            if processed >= max_epoch * oracle.m:
                break

    return x, "success", history


def saga(
    oracle,
    x_0,
    step_size=1e-2,
    batch_size=1,
    max_epoch=100,
    init_table=True,
    trace=False,
    display=False,
    random_state=42,
):
    """
    SAGA для конечной суммы.

    Таблица хранит индивидуальные градиенты loss без L2. Среднее таблицы обновляется
    рекуррентно: mean <- mean + sum(new - old) / m.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x_0, dtype=float).copy()
    table = np.zeros((oracle.m, oracle.n), dtype=float)
    table_mean = np.zeros(oracle.n, dtype=float)
    epochs_spent = 0.0
    next_log_epoch = 1
    start_time = perf_counter()
    history = _init_history(oracle, x, start_time, trace)

    if init_table:
        rows = np.arange(oracle.m)
        table = oracle.individual_loss_grad(x, rows)
        table_mean = table.mean(axis=0)
        epochs_spent = 1.0
        if epochs_spent >= next_log_epoch:
            _record(history, oracle, x, next_log_epoch, start_time)
            next_log_epoch += 1

    while epochs_spent < max_epoch:
        idx = rng.choice(oracle.m, size=min(batch_size, oracle.m), replace=False)
        new_grads = oracle.individual_loss_grad(x, idx)
        old_grads = table[idx].copy()
        correction = (new_grads - old_grads).mean(axis=0)
        grad_est = correction + table_mean + oracle.regcoef * x

        x -= step_size * grad_est
        table[idx] = new_grads
        table_mean += np.sum(new_grads - old_grads, axis=0) / oracle.m

        epochs_spent += idx.size / oracle.m
        if epochs_spent >= next_log_epoch:
            _record(history, oracle, x, next_log_epoch, start_time)
            if display:
                print(f"SAGA epoch={next_log_epoch}, F={history['func'][-1]:.6e}")
            next_log_epoch += 1

    return x, "success", history

