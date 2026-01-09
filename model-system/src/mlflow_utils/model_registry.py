"""
Docstring for model-system.src.mlflow_utils.model_registry
"""
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from loguru import logger

class ModelRegistry:
    def __init__(self, tracking_uri: str):
        self.client = MlflowClient(tracking_uri=tracking_uri)
        logger.info(f"Initialized Model Registry: {tracking_uri}")
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: dict[str,str] | None = None,
        description: str | None = None
    ) -> ModelVersion:
        """
        Register a model from a run
        Args:
            model_uri: URI of model (e.g., "runs:/<run_id>/model")
            model_name: Name to register model under
            tags: Tags to add to model version
            description: Description of model version
            
        Returns:
            ModelVersion object
        """

        logger.info(f"Registering model: {model_name}")
        logger.info(f"Model URI: {model_uri}")

        try:
            self.client.get_registered_model(model_name)
        except Exception:
            logger.info(f"Registered model '{model_name}' not found. Creating it.")
            self.client.create_registered_model(
                name=model_name,
                description=description,
            )

        model_version = self.client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=model_uri.split("/")[1] if "runs:" in model_uri else None,
            description=description,
        )

        if tags:
            for key, val in tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=val
                )
        
        logger.info(f"Model registered: {model_name} v{model_version.version}")
        return model_version
    

    def create_registered_model(
        self,
        name: str,
        tags: dict[str,str] | None = None,
        description: str | None = None
    ): 
        """
        Create a new registered model
        """
        try:
            self.client.create_registered_model(
                name=name,
                tags=tags,
                description=description
            )
        except Exception as e:
            logger.warning(f"Model {name=} may already exist: {e}")
        
    def set_model_version_alias(
        self,
        model_name: str,
        version: str,
        alias: str,
    ):
        """
        Set an alias for a model version (e.g., "champion", "staging")
        
        Args:
            alias: Alias name (e.g., "champion", "staging", "production")
        """
        logger.info(f"Setting alias '{alias}' for {model_name} v{version}")
        
        self.client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version,
        )
        
        logger.info(f"Alias set: {model_name}@{alias} -> v{version}")

    def delete_model_version_alias(
        self,
        model_name: str,
        alias: str,
    ):
        """
        Delete an alias
        
        Args:
            model_name: Name of registered model
            alias: Alias to delete
        """
        self.client.delete_registered_model_alias(
            name=model_name,
            alias=alias,
        )
        logger.info(f"Deleted alias: {model_name}@{alias}")
    
    def get_model_version_by_alias(
        self,
        model_name: str,
        alias: str,
    ) -> ModelVersion:
        """
        Get model version by alias
        
        Args:
            model_name: Name of registered model
            alias: Alias name
            
        Returns:
            ModelVersion object
        """
        return self.client.get_model_version_by_alias(
            name=model_name,
            alias=alias,
        )

    def get_latest_versions(
        self,
        model_name: str,
        stages: list[str] | None = None,
    ) -> list[ModelVersion]:
        """
        Get latest versions of a model
        
        Args:
            model_name: Name of registered model
            stages: List of stages to filter (None = all stages)
            
        Returns:
            List of ModelVersion objects
        """
        return self.client.get_latest_versions(
            name=model_name,
            stages=stages,
        )
    
    def search_model_versions(
        self,
        filter_string: str = "",
        max_results: int = 100,
    ) -> list[ModelVersion]:
        """
        Search model versions
        
        Args:
            filter_string: Filter string (e.g., "name='my_model'")
            max_results: Maximum results to return
            
        Returns:
            List of ModelVersion objects
        """
        return self.client.search_model_versions(
            filter_string=filter_string,
            max_results=max_results,
        )
    
    def transition_model_version_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = True,
    ):
        """
        Transition model version to a stage
        
        Args:
            model_name: Name of registered model
            version: Version number
            stage: Target stage ("Staging", "Production", "Archived")
            archive_existing_versions: Archive other versions in target stage
        """
        logger.info(
            f"Transitioning {model_name} v{version} to {stage}"
        )
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )
        
        logger.info(f"Model transitioned to {stage}")

    def delete_model_version(
        self,
        model_name: str,
        version: str,
    ):
        """
        Delete a model version
        
        Args:
            model_name: Name of registered model
            version: Version number
        """
        self.client.delete_model_version(
            name=model_name,
            version=version,
        )
        logger.info(f"Deleted {model_name} v{version}")
    
    def get_model_info(
        self,
        model_name: str,
    ) -> dict:
        """
        Get model information including all versions
        
        Args:
            model_name: Name of registered model
            
        Returns:
            Dictionary with model info
        """
        model = self.client.get_registered_model(model_name)
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        info = {
            "name": model.name,
            "description": model.description,
            "creation_timestamp": model.creation_timestamp,
            "last_updated_timestamp": model.last_updated_timestamp,
            "versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id,
                    "creation_timestamp": v.creation_timestamp,
                }
                for v in versions
            ],
        }
        
        return info

    def list_registered_models(
        self,
        max_results: int = 100
    ):
        models = self.client.search_registered_models(max_results=max_results)
        return [model.name for model in models]

    
    def promote_model(
        self,
        model_name: str,
        version: str,
        from_alias: str = "staging",
        to_alias: str = "champion",
    ):
        """
        Promote a model from staging to production
        
        Args:
            model_name: Model name
            version: Version to promote
            from_alias: Source alias
            to_alias: Target alias
        """
        logger.info(
            f"Promoting {model_name} v{version} "
            f"from {from_alias} to {to_alias}"
        )

        try:
            existing = self.get_model_version_by_alias(
                model_name=model_name,
                alias=to_alias,
            )
            logger.info(
                f"Clearing existing alias '{to_alias=}' and '{from_alias=}' "
                f"from v{existing.version}"
            )
            self.delete_model_version_alias(
                model_name=model_name,
                alias=to_alias,
            )

            self.delete_model_version_alias(
                model_name=model_name,
                alias=from_alias,
            )
        except Exception:
            logger.info(
                f"No existing alias '{to_alias}' found — safe to continue"
            )
        
        self.set_model_version_alias(model_name, version, to_alias)
        
        logger.info(f"Model promoted to {to_alias}")
    
    
    