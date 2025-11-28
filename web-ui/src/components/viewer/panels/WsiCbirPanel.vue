<template>
  <div>
    <div class="wsi-retrieval">
      <h1>{{ $t('wsi similarity search') }}</h1>
      <a @click="closeRetrieval()">
        <span class="fas fa-times-circle"></span>
      </a>
    </div>

    <div>
      <h5>{{ $t('Search Parameters') }}</h5>

      <!-- Staining For now unavailable
      <b-field label="Staining">
        <b-input v-model="staining" placeholder="Staining (optional)"></b-input>
      </b-field>


      <b-field label="Organ">
        <b-input v-model="organ" placeholder="Organ (optional)"></b-input>
      </b-field>

    
      <b-field label="Species">
        <b-input v-model="species" placeholder="Species (optional)"></b-input>
      </b-field>


      <b-field label="Diagnosis">
        <b-input v-model="diagnosis" placeholder="Diagnosis (optional)"></b-input>
      </b-field> -->

      <!-- k -->
      <b-field label="k (number of results)">
        <b-input type="number" v-model.number="k"></b-input>
      </b-field>

      <button
        class="button is-small is-fullwidth"
        @click="searchSimilarImages"
        :disabled="isLoading"
        :class="{ 'is-loading': isLoading }"
      >
        {{ $t('Search') }}
      </button>

      <div class="results-container">
        <div
          v-if="similarityRows.length"
          class="results-table-container"
        >
          <table class="table is-fullwidth is-striped is-hoverable">
            <thead>
              <tr>
                <th>#</th>
                <th>Image ID</th>
                <th>Distance</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in similarityRows" :key="row.id">
                <td>{{ row.rank }}</td>
                <td>{{ row.id }}</td>
                <td>{{ row.distance.toFixed(4) }}</td>
                <td>
                  <button
                    class="button is-small"
                    @click="addImage(row.id)"
                  >
                    Display
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <p v-else class="has-text-grey is-italic no-results">
          No results yet.
        </p>
      </div>
    </div>
  </div>
</template>

<script>
import {Cytomine, ImageInstanceCollection} from '@/api';
import {get} from '@/utils/store-helpers';

export default {
  name: 'WsiCbirPanel',
  props: {
    index: String,
  },
  data() {
    return {
      data: null,       // will store the raw response from the service
      isLoading: false, // controls button disabled + spinner

      // filters / inputs
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
    project: get('currentProject/project'),
    image() {
      return this.viewerWrapper.images[this.index].imageInstance;
    },
    viewerModule() {
      return this.$store.getters['currentProject/currentViewerModule'];
    },
    // Build nice rows from the similarities, skipping index 0
    similarityRows() {
      if (!this.data || !Array.isArray(this.data.similarities)) {
        return [];
      }

      // similarities is of form: [ [ "IMAGE_id", distance ], ... ]
      // skip the first element (the query image itself)
      return this.data.similarities.slice(1).map(([id, distance], idx) => ({
        rank: idx + 1,
        id,
        distance,
      }));
    },
  },
  methods: {
    async searchSimilarImages() {
      console.log('invoking search');
      const params = {
        query: this.image.baseImage, 
        datasets: '', // TODO: Rename to stores/storages in all affected components
        staining: this.staining || '', // TODO: Exchange & to AND so it doesnt break HTTP and then reverse in microservice
        organ: this.organ || '',
        species: this.species || '',
        diagnosis: this.diagnosis || '',
        k: this.k || 3,
      };

      this.isLoading = true;
      this.data = null;

      try {
        const response = await Cytomine.instance.api.get('wsi-cbir/retrieval', {
          params,
        });
        this.data = response.data;
      } catch (e) {
        console.error('Error while searching similar images:', e);
      } finally {
        this.isLoading = false;
      }
    },
    async findImage(name) {
      let images = (await ImageInstanceCollection.fetchAll({
        filterKey: 'project',
        filterValue: this.project.id,
      })).array;
      return images.find(img => img.originalFilename === name);
    },
    async addImage(imageId) {
      console.log('trying to load another image');
      // try {
      await this.image.fetch(); // refetch image to ensure we have latest version
      let slice = await this.image.fetchReferenceSlice();
      let imgToLoad = await this.findImage(imageId);
      await this.$store.dispatch(this.viewerModule + 'addImage', {imgToLoad, slices: [slice]});
      // } catch (error) {
      //   console.log(error);
      //   this.$notify({type: 'error', text: this.$t('notif-error-add-viewer-image')});
      // }
    },
  },
};
</script>

<style scoped>
.wsi-retrieval {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.results-container {
  margin-top: 1rem;
}

.results-table-container {
  max-height: 300px;
  overflow-y: auto;
  margin-top: 0.5rem;
}

.no-results {
  margin-top: 1rem;
}

/* You can remove this if you no longer use the textarea */
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
