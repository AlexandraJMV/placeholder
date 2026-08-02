# =============================================================================
# VERIFICACIÓN EMPÍRICA DE WEIGHT SHARING EN LA SUPERRED
# =============================================================================
# Objetivo: probar, no asumir, que dos paths distintos que comparten un bloque
# o un stitch usan literalmente los MISMOS tensores de parámetros — no copias,
# no clones, no instancias distintas con valores iguales.
#
# Tres niveles de evidencia, de más a menos directa:
#   1. Identidad de objeto (`is`)      → prueba estructural, 100% concluyente
#   2. Identidad de memoria (data_ptr) → prueba a nivel de tensor, 100% concluyente
#   3. Propagación de gradiente cruzada → prueba funcional/dinámica
# =============================================================================
import torch
from simple_poc.supernet import SuperNetwork


def verify_module_identity(model: SuperNetwork, path_a: list, path_b: list):
    """
    Nivel 1+2: para cada frontera de stages, si path_a y path_b coinciden en
    los índices relevantes, el módulo (stage o stitch) usado DEBE ser el mismo
    objeto en memoria. Se verifica con `is` (identidad de objeto Python) y con
    `data_ptr()` (identidad de la memoria subyacente del tensor).
    """
    print("═" * 70)
    print(f"Path A: {path_a}")
    print(f"Path B: {path_b}")
    print("═" * 70)

    all_ok = True

    # --- Stages ---
    for i in range(model.num_stages):
        mod_a = model.stages[i][path_a[i]]
        mod_b = model.stages[i][path_b[i]]
        same_choice = path_a[i] == path_b[i]
        is_same_obj = mod_a is mod_b

        status = "✅" if (is_same_obj == same_choice) else "❌ INCONSISTENTE"
        print(f"  Stage {i}: path_a[{path_a[i]}] vs path_b[{path_b[i]}]  "
              f"→ mismo módulo: {is_same_obj}  (esperado: {same_choice})  {status}")

        if is_same_obj:
            # Verificación a nivel de tensor: los parámetros deben apuntar a la
            # misma dirección de memoria, no solo tener valores iguales.
            params_a = list(mod_a.parameters())
            params_b = list(mod_b.parameters())
            ptrs_match = all(
                pa.data_ptr() == pb.data_ptr() for pa, pb in zip(params_a, params_b)
            )
            print(f"           data_ptr() de todos los parámetros coincide: {ptrs_match}")
            all_ok &= ptrs_match

        all_ok &= (is_same_obj == same_choice)

    # --- Stitches ---
    for i in range(model.num_stages - 1):
        src_a, dst_a = path_a[i], path_a[i + 1]
        src_b, dst_b = path_b[i], path_b[i + 1]
        same_edge = (src_a == src_b) and (dst_a == dst_b)

        stitch_a = model.stitches[i][src_a][dst_a]
        stitch_b = model.stitches[i][src_b][dst_b]
        is_same_obj = stitch_a is stitch_b

        status = "✅" if (is_same_obj == same_edge) else "❌ INCONSISTENTE"
        print(f"  Stitch {i}→{i+1}: edge_a=({src_a}→{dst_a}) vs edge_b=({src_b}→{dst_b})  "
              f"→ mismo módulo: {is_same_obj}  (esperado: {same_edge})  {status}")

        if is_same_obj:
            params_a = list(stitch_a.parameters())
            params_b = list(stitch_b.parameters())
            ptrs_match = all(
                pa.data_ptr() == pb.data_ptr() for pa, pb in zip(params_a, params_b)
            )
            print(f"           data_ptr() de todos los parámetros coincide: {ptrs_match}")
            all_ok &= ptrs_match

        all_ok &= (is_same_obj == same_edge)

    print("─" * 70)
    print("✅ TODO CONSISTENTE — comparten módulos exactamente donde deberían" if all_ok
          else "❌ HAY INCONSISTENCIAS — revisar construcción de la supernet")
    print("═" * 70)
    return all_ok


def verify_cross_path_gradient(model: SuperNetwork, path_a: list, path_b: list, device="cpu"):
    """
    Nivel 3: prueba dinámica. Si un stitch/bloque es realmente compartido,
    entonces entrenar con path_a debe dejar gradiente en ese módulo, y
    entrenar con path_b (que también lo usa) debe ACUMULAR gradiente en el
    MISMO tensor .grad — no reemplazarlo, no dejarlo intacto.

    Esta versión compara los tensores completos (no solo su norma) y calcula
    el delta explícitamente, para evitar falsos positivos/negativos por
    redondeo o coincidencias en la norma agregada.
    """
    model = model.to(device)
    model.train()
    model.set_bn_tracking(False)

    shared_edge = None
    for i in range(model.num_stages - 1):
        if path_a[i] == path_b[i] and path_a[i + 1] == path_b[i + 1]:
            shared_edge = (i, path_a[i], path_a[i + 1])
            break

    if shared_edge is None:
        print("⚠️  path_a y path_b no comparten ningún stitch — elige paths con al "
              "menos una transición idéntica para esta prueba.")
        return

    i, a_idx, b_idx = shared_edge
    shared_module = model.stitches[i][a_idx][b_idx]
    target_param  = next(shared_module.parameters())
    print(f"\nVerificando gradiente cruzado en stitch compartido: stage {i}→{i+1}, "
          f"edge ({a_idx}→{b_idx})")
    print(f"  Parámetro objetivo: shape={tuple(target_param.shape)}, "
          f"id(tensor)={id(target_param)}, requires_grad={target_param.requires_grad}")

    # --- Paso 1: SOLO path_a, desde cero ---
    model.zero_grad(set_to_none=True)
    x1 = torch.randn(2, 3, model.input_size, model.input_size, device=device)
    model(x1, path=path_a).sum().backward()

    assert target_param.grad is not None, \
        "❌ El gradiente es None tras backward — el módulo NO participó en el forward. Bug real."
    grad_after_a = target_param.grad.detach().clone()
    print(f"  [1] Tras path_a  → grad.norm() = {grad_after_a.norm().item():.6f}")

    # --- Paso 2: SOLO path_b, desde cero (grad limpio) — mide su aporte individual ---
    model.zero_grad(set_to_none=True)
    x2 = torch.randn(2, 3, model.input_size, model.input_size, device=device)
    model(x2, path=path_b).sum().backward()

    assert target_param.grad is not None, \
        "❌ El gradiente es None tras backward con path_b — el módulo NO participó. Bug real."
    grad_only_b = target_param.grad.detach().clone()
    print(f"  [2] Tras path_b (grad limpio) → grad.norm() = {grad_only_b.norm().item():.6f}")

    # --- Paso 3: path_a seguido de path_b SIN limpiar — mide si se acumula ---
    model.zero_grad(set_to_none=True)
    model(x1, path=path_a).sum().backward()
    model(x2, path=path_b).sum().backward()   # sin zero_grad entre medio
    grad_accumulated = target_param.grad.detach().clone()
    print(f"  [3] path_a → path_b SIN zero_grad → grad.norm() = {grad_accumulated.norm().item():.6f}")

    # --- Verificación matemática: [3] debe ser aprox. igual a [1] + [2] ---
    expected_sum = grad_after_a + grad_only_b
    matches_expected = torch.allclose(grad_accumulated, expected_sum, atol=1e-5)
    identical_to_a_alone = torch.equal(grad_accumulated, grad_after_a)

    print(f"\n  ¿[3] ≈ [1] + [2] (acumulación real)?        {matches_expected}")
    print(f"  ¿[3] es idéntico a [1] (NO se acumuló — mal)? {identical_to_a_alone}")

    if matches_expected and not identical_to_a_alone:
        print("  ✅ Gradiente se acumula correctamente entre paths que comparten el módulo.")
    else:
        print("  ❌ Algo está mal — revisar si el módulo realmente participa en ambos forwards, "
              "o si hay un `retain_graph`/`detach` interfiriendo en el camino.")


if __name__ == "__main__":
    PLAN_PATH  = "network_plan.pkl"   # ajusta a tu ruta real
    INPUT_SIZE = 160

    model = SuperNetwork(PLAN_PATH, input_size=INPUT_SIZE)

    # Dos paths que comparten stage 0 y la transición 0→1, pero divergen después
    path_a = [0, 1, 2, 0]
    path_b = [0, 1, 0, 2]

    verify_module_identity(model, path_a, path_b)
    verify_cross_path_gradient(model, path_a, path_b)

    # Bonus: cuenta de parámetros — si NO hubiera sharing, 81 (o 650) paths
    # entrenados independientemente necesitarían ~N_paths × params_por_path.
    # Con sharing real, el total de parámetros de la supernet es órdenes de
    # magnitud menor que esa suma.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total de parámetros en la supernet completa: {total_params:,}")
    print(f"   (compara esto contra: N_paths × params_promedio_por_path,")
    print(f"    que sería el total SIN sharing — debería ser muchísimo mayor)")