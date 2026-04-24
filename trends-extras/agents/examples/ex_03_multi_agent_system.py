"""
Multi-Agent System - Agentes Especializados que Colaboran

Sistema con múltiples agentes que tienen expertise distintas y colaboran.
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class AgentRole(Enum):
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    COORDINATOR = "coordinator"

@dataclass
class Message:
    """Mensaje entre agentes"""
    sender: str
    recipient: str
    content: str
    type: str  # "request", "response", "broadcast"

class Agent:
    """Agente base con capacidades de comunicación"""

    def __init__(self, name: str, role: AgentRole, expertise: str):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.inbox: List[Message] = []
        self.memory: List[Dict] = []

    def receive_message(self, message: Message):
        """Recibe mensaje de otro agente"""
        self.inbox.append(message)
        print(f"📨 {self.name} recibió: {message.content[:50]}...")

    def process_task(self, task: str) -> str:
        """Procesa tarea según expertise"""
        raise NotImplementedError

    def send_message(self, recipient: str, content: str, msg_type: str = "response") -> Message:
        """Envía mensaje a otro agente"""
        message = Message(
            sender=self.name,
            recipient=recipient,
            content=content,
            type=msg_type
        )
        print(f"📤 {self.name} → {recipient}: {content[:50]}...")
        return message

class ResearcherAgent(Agent):
    """Agente especializado en búsqueda de información"""

    def process_task(self, task: str) -> str:
        print(f"\n🔍 {self.name} investigando: {task}")

        # Simulación de research
        research_results = {
            "ai agents": "AI agents son sistemas autónomos que usan LLMs para razonar y actuar. Arquitectura: perception → reasoning → action. Frameworks principales: LangChain, AutoGPT.",
            "multi-agent": "Multi-agent systems coordinan múltiples agentes especializados. Beneficios: especialización, paralelización, robustez.",
            "langchain": "LangChain es framework para construir aplicaciones con LLMs. Incluye agents, chains, memory, tools."
        }

        for key, value in research_results.items():
            if key in task.lower():
                self.memory.append({"task": task, "result": value})
                return value

        return f"Investigué sobre '{task}' pero necesito más contexto."

class AnalystAgent(Agent):
    """Agente especializado en análisis de datos"""

    def process_task(self, task: str) -> str:
        print(f"\n📊 {self.name} analizando: {task}")

        # Simulación de análisis
        if "comparar" in task.lower():
            analysis = """
            Análisis comparativo:
            - LangChain: Mejor para prototyping rápido, ecosystem rico
            - AutoGPT: Más autónomo, menos control
            - Custom: Control total, más trabajo inicial

            Recomendación: LangChain para MVPs, custom para producción a escala
            """
        elif "pros" in task.lower() or "cons" in task.lower():
            analysis = """
            Ventajas: Automatización, escalabilidad, especialización
            Desventajas: Complejidad, costo, potenciales hallucinations
            """
        else:
            analysis = f"Análisis de '{task}': Requiero más datos específicos."

        self.memory.append({"task": task, "result": analysis})
        return analysis

class WriterAgent(Agent):
    """Agente especializado en síntesis y escritura"""

    def process_task(self, task: str) -> str:
        print(f"\n✍️  {self.name} sintetizando: {task}")

        # Compilar información de memoria de otros agentes
        # En sistema real, accedería a mensajes recibidos

        report = f"""
# Reporte sobre {task}

Basado en investigación y análisis del equipo, aquí está la síntesis:

{task}

Este reporte fue generado colaborativamente por el multi-agent system.
"""

        self.memory.append({"task": task, "result": report})
        return report


class CoordinatorAgent(Agent):
    """Agente coordinador que orquesta el trabajo del equipo"""

    def __init__(
            self, name: str, team: Dict[str, Agent]):
        super().__init__(
            name,
            AgentRole.COORDINATOR,
            "coordination and task decomposition"
        )
        self.team = team

    def process_task(self, task: str) -> str:
        print(f"\n🎯 {self.name} coordinando tarea: {task}\n")

        # Descomponer tarea en subtareas
        print("📋 Descomponiendo tarea en subtareas...")
        subtasks = self._decompose_task(task)

        results = {}

        # Asignar subtareas a agentes especializados
        for subtask, agent_role in subtasks:
            agent = self._get_agent_by_role(agent_role)
            if agent:
                # Enviar tarea
                message = self.send_message(
                    recipient=agent.name,
                    content=subtask,
                    msg_type="request"
                )
                agent.receive_message(message)

                # Procesar y guardar resultado
                result = agent.process_task(subtask)
                results[subtask] = result

                # Recibir respuesta
                response = agent.send_message(
                    recipient=self.name,
                    content=result[:100] + "...",
                    msg_type="response"
                )
                self.receive_message(response)

        # Compilar reporte final
        print("\n📝 Compilando reporte final...")
        writer = self._get_agent_by_role(AgentRole.WRITER)
        if writer:
            summary = "\n\n".join([f"**{task}**:\n{result}" for task, result in results.items()])
            final_report = writer.process_task(summary)
            return final_report

        return "Tarea completada pero no se pudo generar reporte."

    def _decompose_task(self, task: str) -> List[tuple]:
        """Descompone tarea en subtareas + agente asignado"""
        # Lógica simple de descomposición
        if "reporte" in task.lower() or "analizar" in task.lower():
            return [
                ("Investigar conceptos principales de " + task, AgentRole.RESEARCHER),
                ("Analizar pros y cons de " + task, AgentRole.ANALYST),
                ("Comparar frameworks mencionados", AgentRole.ANALYST),
            ]
        else:
            return [
                ("Investigar " + task, AgentRole.RESEARCHER),
                ("Analizar " + task, AgentRole.ANALYST),
            ]

    def _get_agent_by_role(self, role: AgentRole) -> Agent:
        """Encuentra agente por rol"""
        for agent in self.team.values():
            if agent.role == role:
                return agent
        return None


# Demo: Multi-Agent System en acción
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 MULTI-AGENT SYSTEM DEMO")
    print("=" * 60)

    # Crear equipo de agentes
    team = {
        "researcher": ResearcherAgent("Alice", AgentRole.RESEARCHER, "information gathering"),
        "analyst": AnalystAgent("Bob", AgentRole.ANALYST, "data analysis"),
        "writer": WriterAgent("Carol", AgentRole.WRITER, "synthesis and writing"),
    }

    # Crear coordinador
    coordinator = CoordinatorAgent("Director", team)

    # Tarea compleja que requiere colaboración
    task = "Crear reporte sobre AI Agents: qué son, frameworks disponibles, y casos de uso"

    # Ejecutar
    final_report = coordinator.process_task(task)

    print("\n\n" + "=" * 60)
    print("📄 REPORTE FINAL")
    print("=" * 60)
    print(final_report)

    # Mostrar memoria de cada agente
    print("\n\n" + "=" * 60)
    print("🧠 MEMORIA DE AGENTES")
    print("=" * 60)
    for name, agent in team.items():
        print(f"\n{agent.name} ({agent.role.value}):")
        print(f"  Tareas procesadas: {len(agent.memory)}")
        print(f"  Mensajes recibidos: {len(agent.inbox)}")

"""
Output esperado:

==============================================================
🤖 MULTI-AGENT SYSTEM DEMO
==============================================================

🎯 Director coordinando tarea: Crear reporte sobre AI Agents...

📋 Descomponiendo tarea en subtareas...
📤 Director → Alice: Investigar conceptos principales...
📨 Alice recibió: Investigar conceptos principales...

🔍 Alice investigando: Investigar conceptos principales...
📤 Alice → Director: AI agents son sistemas autónomos...
📨 Director recibió: AI agents son sistemas autónomos...

[... más interacciones ...]

📝 Compilando reporte final...

✍️  Carol sintetizando: ...

==============================================================
📄 REPORTE FINAL
==============================================================
# Reporte sobre [task]
...

==============================================================
🧠 MEMORIA DE AGENTES
==============================================================
Alice (researcher):
  Tareas procesadas: 1
  Mensajes recibidos: 1
...
"""
