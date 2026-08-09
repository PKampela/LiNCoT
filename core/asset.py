# asset metadata class for storing information about assets in the session

from uuid import UUID, uuid4
from pathlib import Path

from dataclasses import dataclass, field


@dataclass
class AssetMetadata:
    """Metadata for an asset in the session."""

    source_path: Path | None = None

    linked: bool = True

    project_path: Path | None = None

    imported_by: str = "user"

    id: UUID = field(default_factory=uuid4)

    checksum: str | None = None

    def to_dict(self) -> dict:
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "linked": self.linked,
            "project_path": str(self.project_path) if self.project_path else None,
            "imported_by": self.imported_by,
            "id": str(self.id),
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AssetMetadata":
        return cls(
            source_path=Path(data["source_path"]) if data["source_path"] else None,
            linked=data.get("linked", True),
            project_path=Path(data["project_path"]) if data.get("project_path") else None,
            imported_by=data.get("imported_by", "user"),
            id=UUID(data["id"]) if "id" in data else uuid4(),
            checksum=data.get("checksum"),
        )

    