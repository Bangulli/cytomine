<template>
  <div>
    <div class="wsi-retrieval">
      <h1>{{ $t('wsi-retrieval') }}</h1>
      <a @click="closeRetrieval()">
        <span class="fas fa-times-circle"></span>
      </a>
    </div>

    <div>
      <h5>{{ $t('similar-images') }}</h5>

      <!-- Query input 
      <b-field label="Query">
        <b-input v-model="query" placeholder="Enter query text"></b-input>
      </b-field> -->

      <!-- Staining -->
      <b-field label="Staining">
        <b-input v-model="staining" placeholder="Staining (optional)"></b-input>
      </b-field>

      <!-- Organ -->
      <b-field label="Organ">
        <b-input v-model="organ" placeholder="Organ (optional)"></b-input>
      </b-field>

      <!-- Species -->
      <b-field label="Species">
        <b-input v-model="species" placeholder="Species (optional)"></b-input>
      </b-field>

      <!-- Diagnosis -->
      <b-field label="Diagnosis">
        <b-input v-model="diagnosis" placeholder="Diagnosis (optional)"></b-input>
      </b-field>

      <!-- k -->
      <b-field label="k (number of results)">
        <b-input type="number" v-model.number="k"></b-input>
      </b-field>

      <button class="button is-small is-fullwidth" @click="searchSimilarImages">
        {{ $t('search-similar-image') }}
      </button>

      <div class="results-container">
        <textarea :value="output" readonly class="results-textfield"></textarea>
      </div>
    </div>
  </div>
</template>

<script>
import {Cytomine} from '@/api';

export default {
  name: 'WsiCbirPanel',
  props: {
    index: String,
  },
  data() {
    return {
      data: null,
      output: '',

      // Added input fields
      staining: '',
      organ: '',
      species: '',
      diagnosis: '',
      k: 3,
    };
  },
  computed: {
    viewerWrapper() {
      return this.$store.getters['currentProject/currentViewer'];
    },
    image() {
      return this.viewerWrapper.images[this.index].imageInstance;
    },
  },
  methods: {
    async searchSimilarImages() {
      if (!this.image) {
        this.output = 'Error: Image not found';
        return;
      }

      console.log('invoking search');
      const params = {
        query: this.image.filename,
        datasets: '',
        staining: this.staining || '',
        organ: this.organ || '',
        species: this.species || '',
        diagnosis: this.diagnosis || '',
        k: this.k || 3,
      };

      // Log request data
      this.output = `=== REQUEST ===\n${JSON.stringify(params, null, 2)}\n\n=== RESPONSE ===\n`;

      this.data = (
        await Cytomine.instance.api.get('wsi-cbir/retrieval', {
          params,
        })
      ).data;

      // Append response data
      this.output += JSON.stringify(this.data, null, 2);
    },
  },
};
</script>

<style scoped>
.results-container {
  margin-top: 1rem;
}

.results-textfield {
  width: 100%;
  height: 300px;
  padding: 0.5rem;
  font-family: monospace;
  font-size: 0.85rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  resize: vertical;
  overflow: auto;
  background-color: #f5f5f5;
}
</style>
