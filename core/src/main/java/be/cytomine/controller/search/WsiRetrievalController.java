package be.cytomine.controller.search;

import jakarta.persistence.EntityManager;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

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

    @GetMapping("/retrieval/retrieval")
    public ResponseEntity<SearchResponse> retrieveSimilarImages(
        @RequestParam(value = "k_best") Long k_best
    ) {
        log.debug("Retrieve similar images for query image");

        return retrievalService.retrieveSimilarImages(k_best);
    }
}