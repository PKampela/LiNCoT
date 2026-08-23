# Project metadata management module
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4


@dataclass
class ProjectMetadata:
    """Metadata describing a LiNCoT project."""

    project_id: UUID = field(default_factory=uuid4)

    name: str = "Untitled Project"

    application: str = "LiNCoT"

    project_version: int = 1

    created: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    modified: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    is_dirty: bool = False

    description: str | None = None

    project_path: Path | None = None

    def to_dict(self) -> dict:
        return {
            "project_id": str(self.project_id),
            "name": self.name,
            "application": self.application,
            "project_version": self.project_version,
            "created": self.created,
            "modified": self.modified,
            "description": self.description,
            "project_path": str(self.project_path) if self.project_path else None,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            project_id=UUID(data["project_id"]),
            name=data["name"],
            application=data.get("application", "LiNCoT"),
            project_version=data.get("project_version", 1),
            created=data.get("created"),
            modified=data.get("modified"),
            description=data.get("description"),
            project_path=Path(data["project_path"]) if data.get("project_path") else None,
        )