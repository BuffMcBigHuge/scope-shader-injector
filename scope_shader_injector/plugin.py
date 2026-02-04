"""Plugin registration for Scope Shader Injector."""

from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    """Register the shader injector pipeline with Scope."""
    from .pipelines.pipeline import ShaderInjectorPipeline

    register(ShaderInjectorPipeline)
