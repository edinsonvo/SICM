"""Gestión de escenarios del SICM v2.0.

La clase :class:`ScenarioManager` guarda, carga, compara y elimina
escenarios de simulación en un archivo JSON persistente.
"""

import json
import os
from datetime import datetime

DEFAULT_STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "reports", "scenarios.json")


class ScenarioManager:
    """Persistencia de escenarios en formato JSON."""

    def __init__(self, store_path=DEFAULT_STORE):
        """Inicializa el gestor con la ruta del archivo de almacenamiento."""
        self.store_path = store_path
        self._ensure_store()

    def _ensure_store(self):
        """Crea el archivo de almacenamiento si no existe."""
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        if not os.path.exists(self.store_path):
            with open(self.store_path, "w", encoding="utf-8") as fh:
                json.dump([], fh, ensure_ascii=False)

    def _read(self):
        """Lee la lista de escenarios del archivo."""
        try:
            with open(self.store_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            return []

    def _write(self, scenarios):
        """Escribe la lista de escenarios en el archivo."""
        with open(self.store_path, "w", encoding="utf-8") as fh:
            json.dump(scenarios, fh, ensure_ascii=False, indent=2)

    def save(self, name, description, model_name, params, before, after,
             policy_type, magnitude):
        """Guarda un escenario y devuelve su identificador."""
        scenarios = self._read()
        scenario = {
            "id": len(scenarios) + 1,
            "name": name,
            "description": description,
            "model": model_name,
            "params": params,
            "before": before,
            "after": after,
            "policy_type": policy_type,
            "magnitude": magnitude,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        scenarios.append(scenario)
        self._write(scenarios)
        return scenario

    def list_all(self):
        """Devuelve todos los escenarios guardados (los más recientes primero)."""
        return list(reversed(self._read()))

    def get(self, scenario_id):
        """Devuelve un escenario por su identificador (o ``None``)."""
        for sc in self._read():
            if sc["id"] == scenario_id:
                return sc
        return None

    def delete(self, scenario_id):
        """Elimina un escenario por su identificador."""
        scenarios = self._read()
        remaining = [sc for sc in scenarios if sc["id"] != scenario_id]
        self._write(remaining)

    def delete_all(self):
        """Elimina todos los escenarios."""
        self._write([])

    @staticmethod
    def compare(scenarios):
        """Devuelve un diccionario con la comparación de varios escenarios."""
        summary = []
        for sc in scenarios:
            before, after = sc.get("before") or {}, sc.get("after") or {}
            dY = (after.get("Y", 0.0) - before.get("Y", 0.0)) if before else 0.0
            dP = (after.get("P", 0.0) - before.get("P", 0.0)) if before else 0.0
            dr = (after.get("r", 0.0) - before.get("r", 0.0)) if before else 0.0
            summary.append({
                "ID": sc.get("id"),
                "Escenario": sc.get("name"),
                "Modelo": sc.get("model"),
                "Política": sc.get("policy_type"),
                "Magnitud (%)": round((sc.get("magnitude") or 0.0) * 100, 1),
                "ΔY": round(dY, 2),
                "Δr (p.p.)": round(dr * 100, 2),
                "ΔP": round(dP, 3),
                "Descripción": (sc.get("description") or "")[:60],
            })
        return summary
