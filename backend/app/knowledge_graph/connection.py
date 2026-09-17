"""
app/knowledge_graph/connection.py

Neo4j connection manager and session lifecycle handler.

Manages connection pooling, credential extraction from application Settings,
connectivity health verification, and graceful driver closure on shutdown.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.config import get_settings
from app.core.exceptions import GraphQueryError, KnowledgeGraphError
from app.core.logging import get_logger

logger = get_logger(__name__)


class Neo4jConnectionManager:
    """Manages the lifecycle and connectivity of the Neo4j database driver."""

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self._driver: Optional[Driver] = None

    def get_driver(self) -> Driver:
        """Return the active Neo4j driver or initialize a new one."""
        if self._driver is None:
            try:
                logger.info("Initializing Neo4j driver for %s (db: %s)", self.uri, self.database)
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=15.0,
                )
            except Exception as exc:
                logger.error("Failed to construct Neo4j driver: %s", str(exc))
                raise KnowledgeGraphError(
                    detail=f"Failed to create Neo4j driver for '{self.uri}': {exc}",
                    context={"uri": self.uri, "database": self.database},
                ) from exc
        return self._driver

    def verify_connectivity(self) -> bool:
        """Verify that the database is reachable and credentials are valid."""
        try:
            driver = self.get_driver()
            driver.verify_connectivity()
            return True
        except (ServiceUnavailable, Neo4jError, Exception) as exc:
            logger.warning("Neo4j connectivity verification failed: %s", str(exc))
            return False

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Provide a contextual Neo4j session with automatic closure and error translation."""
        driver = self.get_driver()
        session = driver.session(database=self.database)
        try:
            yield session
        except (ServiceUnavailable, ConnectionError) as exc:
            logger.error("Neo4j service unavailable: %s", str(exc))
            raise KnowledgeGraphError(
                detail=f"Neo4j database is unreachable at {self.uri}: {exc}",
                context={"uri": self.uri, "database": self.database},
            ) from exc
        except Neo4jError as exc:
            logger.error("Neo4j query execution error: %s", str(exc))
            raise GraphQueryError(
                detail=f"Cypher query error: {exc.message}",
                context={"code": exc.code, "message": exc.message},
            ) from exc
        finally:
            session.close()

    def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a read/write Cypher query and return results as list of dicts."""
        params = parameters or {}
        with self.get_session() as session:
            try:
                result = session.run(query, params)
                return [record.data() for record in result]
            except Neo4jError as exc:
                raise GraphQueryError(
                    detail=f"Failed to execute Cypher query: {exc.message}",
                    context={"query": query, "parameters": params, "error_code": exc.code},
                ) from exc

    def close(self) -> None:
        """Close the driver and release all connection pool resources."""
        if self._driver is not None:
            logger.info("Closing Neo4j driver")
            try:
                self._driver.close()
            except Exception as exc:
                logger.warning("Error while closing Neo4j driver: %s", str(exc))
            finally:
                self._driver = None


_cached_manager: Optional[Neo4jConnectionManager] = None


def get_neo4j_manager() -> Neo4jConnectionManager:
    """Singleton getter for Neo4jConnectionManager."""
    global _cached_manager
    if _cached_manager is None:
        _cached_manager = Neo4jConnectionManager()
    return _cached_manager
