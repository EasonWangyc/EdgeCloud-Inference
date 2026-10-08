"""HTTP adapters and composed runtimes release their owned connections."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from parksight_vlm.inference import (
    EdgeCloudRouterRuntime, EdgeLlmHttpBackend, EdgeLlmRuntime, RiskRuntime, RoutingPolicy,
)


class RuntimeCleanupTests(unittest.TestCase):
    def edge(self, backend):
        return EdgeLlmRuntime(
            data_root=Path("."), backend=backend, backend_revision="test",
            model_id="model", model_revision="revision", adapter_revision="none", precision="fp16",
        )

    def router(self, edge, cloud):
        return EdgeCloudRouterRuntime(
            data_root=Path("."), edge_runtime=edge, cloud_runtime=cloud,
            signals_provider=Mock(), policy=RoutingPolicy(), backend_revision="test",
            model_id="model", model_revision="revision",
        )

    def test_edge_runtime_closes_actual_http_backend_connection_idempotently(self):
        backend = EdgeLlmHttpBackend(reuse_http_connection=True)
        connection = Mock()
        backend._http_connection = connection
        runtime = self.edge(backend)
        runtime.close()
        runtime.close()
        connection.close.assert_called_once_with()
        self.assertIsNone(backend._http_connection)

    def test_generate_only_backend_remains_compatible(self):
        backend = SimpleNamespace(generate=Mock())
        self.edge(backend).close()
        backend.generate.assert_not_called()

    def test_router_closes_both_owned_children(self):
        edge, cloud = Mock(spec=RiskRuntime), Mock(spec=RiskRuntime)
        self.router(edge, cloud).close()
        edge.close.assert_called_once_with()
        cloud.close.assert_called_once_with()

    def test_edge_cleanup_failure_still_releases_cloud(self):
        edge, cloud = Mock(spec=RiskRuntime), Mock(spec=RiskRuntime)
        edge.close.side_effect = RuntimeError("edge cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "edge cleanup failed"):
            self.router(edge, cloud).close()
        cloud.close.assert_called_once_with()

    def test_cloud_cleanup_failure_is_reported_after_edge_cleanup(self):
        edge, cloud = Mock(spec=RiskRuntime), Mock(spec=RiskRuntime)
        cloud.close.side_effect = RuntimeError("cloud cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "cloud cleanup failed"):
            self.router(edge, cloud).close()
        edge.close.assert_called_once_with()

    def test_aliased_child_is_closed_once(self):
        child = Mock(spec=RiskRuntime)
        self.router(child, child).close()
        child.close.assert_called_once_with()
