package be.cytomine.controller.search;

import jakarta.persistence.EntityManager;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpClientErrorException;
import java.util.Map;
import be.cytomine.domain.ontology.AnnotationDomain;
import be.cytomine.dto.search.SearchResponse;
import be.cytomine.service.search.RetrievalService;
import be.cytomine.service.search.WsiRetrievalService;

@Slf4j
@RequiredArgsConstructor
@RequestMapping("/api")
@RestController
public class WsiRetrievalController {

    private final WsiRetrievalService retrievalService;

    @PostMapping("/wsi-cbir/retrieval")
    public ResponseEntity<String> retrieveSimilarImages(
        @RequestBody Map<String, Object> body
    ) {
        Long k = ((Number) body.get("k")).longValue();
        Long query = ((Number) body.get("query")).longValue();
        String datasets = (String) body.get("datasets");
        String staining = (String) body.get("staining");
        String organ = (String) body.get("organ");
        String species = (String) body.get("species");
        String diagnosis = (String) body.get("diagnosis");
        String project_id = (String) body.get("projectId");
    
        log.debug("Retrieve similar images for query image");

        return retrievalService.retrieveSimilarImages(k, query, datasets, staining, organ, species, diagnosis, project_id);
    }

    @GetMapping("/wsi-cbir/jobs/{jobId}")
    public ResponseEntity<Map<String, Object>> getJob(
        @PathVariable String jobId
    ) {
        log.debug("Retrieve job {}", jobId);

        try {
            ResponseEntity<SearchResponse> resp = retrievalService.getJob(jobId);
            if (resp.getStatusCode() == HttpStatus.OK) {
                return ResponseEntity.ok(Map.of("state", "DONE", "result", resp.getBody()));
            } else if (resp.getStatusCode() == HttpStatus.ACCEPTED) {
                return ResponseEntity.status(202).body(Map.of("state", "PENDING"));
            } else {
                return ResponseEntity.status(500).body(Map.of("state", "FAILED", "error", "Unexpected status: " + resp.getStatusCode()));
            }
        } catch (HttpClientErrorException e) {
            if (e.getStatusCode() == HttpStatus.NOT_FOUND) {
                return ResponseEntity.status(202).body(Map.of("state", "PENDING"));
            } else {
                return ResponseEntity.status(500).body(Map.of("state", "FAILED", "error", e.getMessage()));
            }
        }
    }

}