"""Project saving and loading management."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path

from core.asset import AssetMetadata
from core.session import Session
from core.image import Image

from typing import Protocol

@dataclass
class ProjectValidationReport:
    """Report generated when validating project assets."""

    missing_assets: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.missing_assets) == 0
    
class ProjectManager:
    """Manage saving, loading, and recovery of TMSLabs projects."""

    PROJECT_FILE = "project.json"

    def save(
        self,
        session: Session,
        project_path: Path,
        bundle_assets: bool = True,
        update_timestamp: bool = True,
    ) -> None:
        """Save a session as a project."""

        project_path.mkdir(parents=True, exist_ok=True)

        session.project.project_path = project_path

        if not session.project.name:
            session.project.name = project_path.name

        session.project.modified = datetime.now().isoformat()

        if bundle_assets:
            self._prepare_asset_directories(project_path)
            self._bundle_assets(session, project_path)

        if update_timestamp:
            session.project.modified = datetime.now().isoformat()

        data = {
            "format_version": "1.0",
            "project": session.project.to_dict(),
            "subject": session.subject.to_dict(),
            "session": session.to_dict(),
        }

        with open(
            project_path / self.PROJECT_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
            )

    def load(self, project_path: Path) -> Session:
        """Load a project from disk."""

        project_file = project_path / self.PROJECT_FILE

        if not project_file.exists():
            raise FileNotFoundError(
                f"Project file not found: {project_file}"
            )

        with open(
            project_file,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if "format_version" not in data:
            raise ValueError(
                "Project file is missing format_version"
            )

        if "session" in data:
            session_data = data["session"]
        else:
            # Compatibility with current project files
            session_data = data

        session = Session.from_dict(session_data)

        report = self._resolve_assets(
            session,
            project_path,
        )

        if not report.valid:
            session.load_warnings = report.missing_assets

        return session

    def default_recovery_path(self) -> Path:
        return (
            Path.home()
            / ".tmslabs"
            / "recovery"
        )

    def autosave(
        self,
        session: Session,
        recovery_path: Path | None = None,
    ) -> None:
        """Save temporary recovery data."""

        if recovery_path is None:
            recovery_path = self.default_recovery_path()

        self.save(
            session,
            recovery_path,
            bundle_assets=False,
            update_timestamp=False,
        )

    def has_recovery(
        self,
        recovery_path: Path | None = None,
    ) -> bool:
        """Check whether an autosave recovery exists."""

        if recovery_path is None:
            recovery_path = self.default_recovery_path()

        return (
            recovery_path
            / self.PROJECT_FILE
        ).exists()

    def clear_recovery(
        self,
        recovery_path: Path | None = None,
    ) -> None:
        """Remove recovery data."""

        if recovery_path is None:
            recovery_path = self.default_recovery_path()

        if recovery_path.exists():
            shutil.rmtree(recovery_path)

    def load_recovery(self) -> Session:
        """Load the default recovery workspace."""

        return self.load(
            self.default_recovery_path()
        )

    def _prepare_asset_directories(
        self,
        project_path: Path,
    ) -> None:
        """Create bundled asset directories."""

        for folder in (
            "images",
            "surfaces",
            "transforms",
        ):
            (
                project_path
                / "assets"
                / folder
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    def _bundle_assets(
        self,
        session: Session,
        project_path: Path,
    ) -> None:
        """Copy assets into the project folder."""

        for image in session.images.values():
            if image.asset:
                self._copy_asset(
                    image.asset,
                    project_path / "assets" / "images",
                    project_path,
                )

        for surface in session.surfaces.values():
            if surface.asset:
                self._copy_asset(
                    surface.asset,
                    project_path / "assets" / "surfaces",
                    project_path,
                )

        for transform in session.transforms.values():
            if transform.asset:
                self._copy_asset(
                    transform.asset,
                    project_path / "assets" / "transforms",
                    project_path,
                )

    def _copy_asset(
        self,
        asset: AssetMetadata,
        destination: Path,
        project_path: Path,
    ) -> None:
        """Copy one asset into the bundled project."""

        if asset.source_path is None:
            return

        source = asset.source_path

        if not source.exists():
            raise FileNotFoundError(
                f"Asset not found: {source}"
            )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        target = destination / source.name

        shutil.copy2(
            source,
            target,
        )

        asset.project_path = target.relative_to(
            project_path
        )

        asset.linked = False

    def _resolve_asset(
        self,
        asset: AssetMetadata,
        project_path: Path,
    ) -> Path:
        """Resolve an asset path when loading a project."""

        if asset.linked:
            if asset.source_path is None:
                raise FileNotFoundError(
                    "Linked asset has no source path"
                )

            return asset.source_path

        if asset.project_path is None:
            raise FileNotFoundError(
                "Bundled asset has no project path"
            )

        return project_path / asset.project_path

    def _resolve_assets(
        self,
        session: Session,
        project_path: Path,
    ) -> ProjectValidationReport:
        """Resolve all assets in a loaded project."""

        report = ProjectValidationReport()

        # ---------------------------------------------------------
        # Images
        # ---------------------------------------------------------

        for name, image in list(session.images.items()):
            if image.asset is None:
                continue

            try:
                path = self._resolve_asset(
                    image.asset,
                    project_path,
                )
            except FileNotFoundError:
                report.missing_assets.append(
                    f"image: {name}"
                )
                continue

            if not path.exists():
                report.missing_assets.append(
                    f"image: {name}"
                )
                continue

            try:
                from backends.nibabel_backend import load_nifti_image

                loaded_data, loaded_affine = load_nifti_image(path)

                loaded_data = np.asarray(loaded_data)
                loaded_affine = np.asarray(
                    loaded_affine,
                    dtype=float,
                )

            except Exception as exc:
                report.missing_assets.append(
                    f"image: {name} (failed to load: {exc})"
                )
                continue

            # -----------------------------------------------------
            # Validate loaded image against saved metadata
            # -----------------------------------------------------

            if tuple(loaded_data.shape) != tuple(image.shape):
                report.missing_assets.append(
                    f"image: {name} (shape mismatch)"
                )
                continue

            if not np.allclose(
                loaded_affine,
                image.affine,
                atol=1e-5,
            ):
                report.missing_assets.append(
                    f"image: {name} (affine mismatch)"
                )
                continue

            # -----------------------------------------------------
            # Construct the fully loaded Image
            # -----------------------------------------------------

            loaded_image = Image(
                data=loaded_data,
                affine=loaded_affine,
                voxel_frame=image.voxel_frame,
                world_frame=image.world_frame,
                asset=image.asset,
            )

            # Replace metadata-only image with loaded image.
            session.images.replace_image(
                name,
                loaded_image,
            )

        # ---------------------------------------------------------
        # Surfaces
        # ---------------------------------------------------------

        for name, surface in session.surfaces.items():
            if surface.asset is None:
                continue

            if not self.validate_asset(
                surface.asset,
                project_path,
            ):
                report.missing_assets.append(
                    f"surface: {name}"
                )

        # ---------------------------------------------------------
        # Transforms
        # ---------------------------------------------------------

        for name, transform in session.transforms.items():
            if transform.asset is None:
                continue

            if not self.validate_asset(
                transform.asset,
                project_path,
            ):
                report.missing_assets.append(
                    f"transform: {name}"
                )

        return report

    def validate_asset(
        self,
        asset: AssetMetadata,
        project_path: Path,
    ) -> bool:
        try:
            path = self._resolve_asset(
                asset,
                project_path,
            )
        except FileNotFoundError:
            return False

        return path.exists()